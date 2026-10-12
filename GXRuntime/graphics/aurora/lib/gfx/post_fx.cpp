#include "post_fx.hpp"

#include <aurora/aurora.h>

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace aurora::gfx::post_fx {
using webgpu::g_device;

namespace {
std::atomic_bool g_bloom{false};
std::atomic_bool g_farBlur{false};
std::atomic_bool g_ao{false};
std::atomic_bool g_clouds{false};
float g_cloudStrength = 0.28f;  // how dark a cloud's shadow is
float g_cloudScale = 7000.f;    // world units across one cloud
bool g_cloudDebug = false;
// The camera, for the cloud shadows (aurora_set_world_view): the inverse view
// matrix's three rows, then tan(fovy / 2), aspect, near, the sea's height, and
// the time in seconds. Written by the game thread, read when a frame is encoded.
std::mutex g_cameraMutex;
std::array<float, 20> g_camera{};
bool g_cameraValid = false;
float g_aoRadius = 30.f;    // the occlusion radius, in near-plane units
float g_aoStrength = 0.6f;  // how dark a fully occluded spot gets
bool g_aoDebug = false;
float g_farStart = 8000.f;  // distance in near-plane units where the far blur starts
float g_farEnd = 30000.f;   // and where it is full
float g_farAmount = 0.8f;  // how much of the blurred scene at full
float g_farBackground = 0.35f;  // and where there is no depth (the sky, the horizon)
bool g_farDebug = false;
float g_threshold = 0.72f;
float g_strength = 0.35f;

wgpu::ShaderModule g_module;
wgpu::BindGroupLayout g_layout;
wgpu::PipelineLayout g_pipelineLayout;
wgpu::Sampler g_sampler;
wgpu::TextureFormat g_format = wgpu::TextureFormat::Undefined;
wgpu::RenderPipeline g_bright, g_blurH, g_blurV;
wgpu::RenderPipeline g_composite;
wgpu::RenderPipeline g_down;           // the scene at a quarter of the size, unfiltered
wgpu::RenderPipeline g_farComposite;   // blended into the EFB by distance
wgpu::RenderPipeline g_aoPass;         // occlusion at a quarter of the size
wgpu::RenderPipeline g_aoApply;        // multiplied into the EFB
uint32_t g_aoSamples = 0;
std::array<wgpu::Texture, 2> g_aoTex;
std::array<wgpu::TextureView, 2> g_aoView;
uint32_t g_farSamples = 0;
wgpu::BindGroupLayout g_depthLayout;
wgpu::PipelineLayout g_depthPipelineLayout;
std::array<wgpu::Texture, 2> g_farTex;
std::array<wgpu::TextureView, 2> g_farView;
uint32_t g_compositeSamples = 0;
std::array<wgpu::Texture, 2> g_tex;
std::array<wgpu::TextureView, 2> g_view;
wgpu::Extent3D g_size{};

constexpr const char* kShader = R"""(
override threshold: f32 = 0.72;
override strength: f32 = 0.35;

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(0.0, 0.0), vec2(0.0, 2.0), vec2(2.0, 0.0));
var<private> weights: array<f32, 5> = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[i], 0.0, 1.0);
    out.uv = uvs[i];
    return out;
}

fn texel() -> vec2<f32> {
    return 1.0 / vec2<f32>(textureDimensions(src));
}

// A quarter of the size: four bilinear taps cover a 4x4 block of the scene.
// Only what is brighter than the threshold passes, with a soft knee.
@fragment
fn fs_bright(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = texel();
    let c = 0.25 * (textureSample(src, samp, in.uv + t * vec2(-1.0, -1.0)).rgb +
                    textureSample(src, samp, in.uv + t * vec2(1.0, -1.0)).rgb +
                    textureSample(src, samp, in.uv + t * vec2(-1.0, 1.0)).rgb +
                    textureSample(src, samp, in.uv + t * vec2(1.0, 1.0)).rgb);
    let luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return vec4(c * smoothstep(threshold, 1.0, luma), 1.0);
}

fn blur(uv: vec2<f32>, dir: vec2<f32>) -> vec4<f32> {
    let step = texel() * dir * 1.5;
    var c = textureSample(src, samp, uv).rgb * weights[0];
    for (var i = 1; i < 5; i++) {
        let o = step * f32(i);
        c += (textureSample(src, samp, uv + o).rgb + textureSample(src, samp, uv - o).rgb) * weights[i];
    }
    return vec4(c, 1.0);
}

@fragment
fn fs_blur_h(in: VertexOutput) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2(1.0, 0.0));
}

@fragment
fn fs_blur_v(in: VertexOutput) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2(0.0, 1.0));
}

@fragment
fn fs_down(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = texel();
    let c = 0.25 * (textureSample(src, samp, in.uv + t * vec2(-1.0, -1.0)).rgb +
                    textureSample(src, samp, in.uv + t * vec2(1.0, -1.0)).rgb +
                    textureSample(src, samp, in.uv + t * vec2(-1.0, 1.0)).rgb +
                    textureSample(src, samp, in.uv + t * vec2(1.0, 1.0)).rgb);
    return vec4(c, 1.0);
}

// Multiplied into the scene (ambient occlusion); the EFB's alpha is kept.
@fragment
fn fs_ao_apply(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(textureSample(src, samp, in.uv).rgb, 1.0);
}

// Added onto the scene; the EFB's alpha is kept.
@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(textureSample(src, samp, in.uv).rgb * strength, 0.0);
}
)""";

// HD's far-distance blur: the scene blurred at a quarter of the size, blended
// back by distance. Aurora's depth is reversed (1 near, 0 far) and, for a
// perspective view, near / distance: 1 / depth is the distance in near-plane
// units. Wind Waker draws the sky, the sea toward the horizon and the islands'
// far models without depth (0): those get the background amount.
constexpr const char* kFarShader = R"""(
override far_start: f32 = 8000.0;
override far_end: f32 = 30000.0;
override amount: f32 = 0.8;
override background: f32 = 0.35;
override debug_view: bool = false;

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var blurred: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(0.0, 0.0), vec2(0.0, 2.0), vec2(2.0, 0.0));

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[i], 0.0, 1.0);
    out.uv = uvs[i];
    return out;
}

@fragment
fn fs_far(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureLoad(depth, vec2<i32>(in.pos.xy), 0);
    // No depth: what the game draws behind the scene without it (the sky, the
    // sea toward the horizon, the islands' far models).
    let f = select(smoothstep(far_start, far_end, 1.0 / max(d, 1e-9)) * amount, background, d <= 0.0);
    if (debug_view) {
        // log2 of the distance as brightness (0 at 1, 1 at 2^24), the sky blue.
        if (d <= 0.0) {
            return vec4(0.0, 0.0, 1.0, 1.0);
        }
        return vec4(vec3(log2(1.0 / d) / 24.0), 1.0);
    }
    return vec4(textureSampleLevel(blurred, samp, in.uv, 0.0).rgb, f);
}
)""";
wgpu::ShaderModule g_farModule;

// Ambient occlusion from depth alone: around each pixel, six pairs of samples
// on opposite sides, in a ring whose size on screen shrinks with distance; a
// pair whose average is nearer the camera than the pixel (a crease) occludes it. Where there is no
// depth (the sky, the horizon) nothing is occluded. Written at a quarter of the
// size, blurred, then multiplied into the scene (fs_ao_apply).
constexpr const char* kAoShader = R"""(
override radius: f32 = 30.0;
override strength: f32 = 0.6;

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var unused: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(0.0, 0.0), vec2(0.0, 2.0), vec2(2.0, 0.0));

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[i], 0.0, 1.0);
    out.uv = uvs[i];
    return out;
}

fn distance_at(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth));
    let p = clamp(vec2<i32>(uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let d = textureLoad(depth, p, 0);
    return select(1.0 / d, 0.0, d <= 0.0);
}

@fragment
fn fs_ao(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = distance_at(in.uv);
    if (dist <= 0.0) {
        return vec4(1.0);
    }
    let size = vec2<f32>(textureDimensions(depth));
    // About a 60-degree view: a radius r at distance z covers 0.87 r / z of the height.
    let reach = vec2(size.y / size.x, 1.0) * (0.87 * radius / dist);
    // Six directions, a sample on each side: a flat surface has its two
    // samples on either side of the pixel's distance, so only a crease (both
    // nearer) occludes. A pair with no depth on a side, or a step much larger
    // than the radius (a silhouette in front of the background), does not.
    var occluded = 0.0;
    for (var i = 0; i < 6; i++) {
        let a = f32(i) * 0.5236 + 0.26;  // 30 degrees apart
        let r = (f32(i % 3) + 1.0) / 3.0;
        let o = vec2(cos(a), sin(a)) * reach * r;
        let n1 = distance_at(in.uv + o);
        let n2 = distance_at(in.uv - o);
        if (n1 > 0.0 && n2 > 0.0) {
            let crease = dist - 0.5 * (n1 + n2);
            occluded += clamp(crease / (0.25 * radius), 0.0, 1.0) * clamp(2.0 - crease / radius, 0.0, 1.0);
        }
    }
    let ao = 1.0 - strength * occluded / 6.0;
    return vec4(vec3(ao), 1.0);
}
)""";
wgpu::ShaderModule g_aoModule;

// Cloud shadows on the sea and the islands: each pixel's place in the world
// (from depth, or where the view ray meets the sea where there is no depth),
// darkened by soft drifting noise in world X/Z that fades with distance.
constexpr const char* kCloudShader = R"""(
override strength: f32 = 0.28;
override scale: f32 = 7000.0;
override debug_view: bool = false;

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var unused: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(0.0, 0.0), vec2(0.0, 2.0), vec2(2.0, 0.0));

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[i], 0.0, 1.0);
    out.uv = uvs[i];
    return out;
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

fn fbm(p0: vec2<f32>) -> f32 {
    var p = p0;
    var v = 0.0;
    var a = 0.5;
    for (var i = 0; i < 4; i++) {
        v += a * value_noise(p);
        p = p * 2.03 + vec2(17.0, 9.0);
        a *= 0.5;
    }
    return v;
}

@fragment
fn fs_clouds(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureLoad(depth, vec2<i32>(in.pos.xy), 0);
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let ray = vec3(ndc.x * view.lens.x * view.lens.y, ndc.y * view.lens.x, -1.0);
    let dir = vec3(dot(view.rows[0].xyz, ray), dot(view.rows[1].xyz, ray), dot(view.rows[2].xyz, ray));
    let eye = vec3(view.rows[0].w, view.rows[1].w, view.rows[2].w);
    var world = eye;
    if (d > 0.0) {
        world = eye + dir * (view.lens.z / d);
    } else {
        // No depth: the sky, or the sea toward the horizon.
        if (dir.y > -1e-4) {
            return vec4(1.0);
        }
        let t = (view.lens.w - eye.y) / dir.y;
        if (t <= 0.0) {
            return vec4(1.0);
        }
        world = eye + dir * t;
    }
    let distance = length(world - eye);
    if (debug_view) {
        // A 1,000-unit world grid in red and green, height in blue (every 500 units).
        let g = fract(world.xz / 1000.0);
        return vec4(g.x, g.y, fract(world.y / 500.0), 1.0);
    }
    let drift = vec2(0.6, 0.25) * view.time.x * 0.02;
    let n = fbm(world.xz / scale + drift);
    let shade = smoothstep(0.42, 0.6, n) * strength * (1.0 - smoothstep(60000.0, 160000.0, distance));
    return vec4(vec3(1.0 - shade), 1.0);
}
)""";
wgpu::ShaderModule g_cloudModule;
wgpu::BindGroupLayout g_cloudLayout;
wgpu::PipelineLayout g_cloudPipelineLayout;
wgpu::RenderPipeline g_cloudPass;
uint32_t g_cloudSamples = 0;

float env_float(const char* name, float fallback, float lo, float hi) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const float parsed = std::strtof(value, &end);
  if (end == value || parsed < lo || parsed > hi) {
    return fallback;
  }
  return parsed;
}

enum Blend { kReplace, kAdd, kMultiply };

wgpu::RenderPipeline make_pipeline(const char* label, const char* entry, wgpu::TextureFormat format,
                                   uint32_t samples, Blend mode) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "threshold", .value = g_threshold},
      wgpu::ConstantEntry{.key = "strength", .value = g_strength},
  };
  const wgpu::BlendState add{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const wgpu::BlendState multiply{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{
      .format = format,
      .blend = mode == kAdd ? &add : mode == kMultiply ? &multiply : nullptr,
      .writeMask = wgpu::ColorWriteMask::All,
  }};
  const wgpu::FragmentState fragment{
      .module = g_module,
      .entryPoint = entry,
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      .layout = g_pipelineLayout,
      .vertex = {.module = g_module, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_ao_pipeline(wgpu::TextureFormat format) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "radius", .value = g_aoRadius},
      wgpu::ConstantEntry{.key = "strength", .value = g_aoStrength},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_aoModule,
      .entryPoint = "fs_ao",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Ambient occlusion",
      .layout = g_depthPipelineLayout,
      .vertex = {.module = g_aoModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_cloud_pipeline(wgpu::TextureFormat format, uint32_t samples) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_cloudStrength},
      wgpu::ConstantEntry{.key = "scale", .value = g_cloudScale},
      wgpu::ConstantEntry{.key = "debug_view", .value = g_cloudDebug ? 1.0 : 0.0},
  };
  const wgpu::BlendState multiply{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = &multiply, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_cloudModule,
      .entryPoint = "fs_clouds",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Cloud shadows",
      .layout = g_cloudPipelineLayout,
      .vertex = {.module = g_cloudModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_far_pipeline(wgpu::TextureFormat format, uint32_t samples) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "far_start", .value = g_farStart},
      wgpu::ConstantEntry{.key = "far_end", .value = g_farEnd},
      wgpu::ConstantEntry{.key = "amount", .value = g_farAmount},
      wgpu::ConstantEntry{.key = "background", .value = g_farBackground},
      wgpu::ConstantEntry{.key = "debug_view", .value = g_farDebug ? 1.0 : 0.0},
  };
  const wgpu::BlendState over{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::SrcAlpha,
                .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = &over, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_farModule,
      .entryPoint = "fs_far",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Far blur composite",
      .layout = g_depthPipelineLayout,
      .vertex = {.module = g_farModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

void ensure_targets(const wgpu::Extent3D& sceneSize, wgpu::TextureFormat format, uint32_t samples) {
  if (format != g_format) {
    g_format = format;
    g_bright = make_pipeline("Bloom bright pass", "fs_bright", format, 1, kReplace);
    g_blurH = make_pipeline("Bloom blur (horizontal)", "fs_blur_h", format, 1, kReplace);
    g_blurV = make_pipeline("Bloom blur (vertical)", "fs_blur_v", format, 1, kReplace);
    g_down = make_pipeline("Far blur downsample", "fs_down", format, 1, kReplace);
    g_aoPass = make_ao_pipeline(format);
    g_cloudPass = {};
    g_composite = {};
    g_farComposite = {};
    g_aoApply = {};
    g_size = {};
  }
  if (!g_composite || samples != g_compositeSamples) {
    g_composite = make_pipeline("Bloom composite", "fs_composite", format, samples, kAdd);
    g_compositeSamples = samples;
  }
  if (!g_cloudPass || samples != g_cloudSamples) {
    g_cloudPass = make_cloud_pipeline(format, samples);
    g_cloudSamples = samples;
  }
  if (!g_aoApply || samples != g_aoSamples) {
    g_aoApply = make_pipeline("Ambient occlusion apply", "fs_ao_apply", format, samples,
                              g_aoDebug ? kReplace : kMultiply);
    g_aoSamples = samples;
  }
  if (!g_farComposite || samples != g_farSamples) {
    g_farComposite = make_far_pipeline(format, samples);
    g_farSamples = samples;
  }
  const wgpu::Extent3D size{std::max(1u, sceneSize.width / 4), std::max(1u, sceneSize.height / 4), 1};
  if (size.width == g_size.width && size.height == g_size.height && g_tex[0]) {
    return;
  }
  g_size = size;
  for (size_t i = 0; i < g_tex.size(); ++i) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Bloom texture",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .dimension = wgpu::TextureDimension::e2D,
        .size = size,
        .format = format,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    g_tex[i] = g_device.CreateTexture(&descriptor);
    g_view[i] = g_tex[i].CreateView();
    g_farTex[i] = g_device.CreateTexture(&descriptor);
    g_farView[i] = g_farTex[i].CreateView();
    g_aoTex[i] = g_device.CreateTexture(&descriptor);
    g_aoView[i] = g_aoTex[i].CreateView();
  }
}

void draw(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
          const wgpu::TextureView& source, const wgpu::TextureView& target, bool load,
          const wgpu::TextureView& resolve = nullptr) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = source},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_layout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  const std::array attachments{wgpu::RenderPassColorAttachment{
      .view = target,
      .resolveTarget = resolve,
      .loadOp = load ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  }};
  const wgpu::RenderPassDescriptor descriptor{
      .label = label,
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };
  const auto pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bindGroup, 0, nullptr);
  pass.Draw(3);
  pass.End();
}
void draw_depth(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                const wgpu::TextureView& source, const wgpu::TextureView& depthView, const wgpu::TextureView& target,
                const wgpu::TextureView& resolve, bool load) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = source},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_depthLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  const std::array attachments{wgpu::RenderPassColorAttachment{
      .view = target,
      .resolveTarget = resolve,
      .loadOp = load ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {1.0, 1.0, 1.0, 1.0},
  }};
  const wgpu::RenderPassDescriptor descriptor{
      .label = label,
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };
  const auto pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bindGroup, 0, nullptr);
  pass.Draw(3);
  pass.End();
}
void draw_clouds(const wgpu::CommandEncoder& cmd, const std::array<float, 20>& view, const wgpu::TextureView& depthView,
                 const wgpu::TextureView& target, const wgpu::TextureView& resolve) {
  // A small buffer for each frame, so a frame still in flight keeps its camera.
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Cloud shadows view",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(view),
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, sizeof(view)), view.data(), sizeof(view));
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = g_farView[0]},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .buffer = buffer, .offset = 0, .size = sizeof(view)},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_cloudLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  const std::array attachments{wgpu::RenderPassColorAttachment{
      .view = target,
      .resolveTarget = resolve,
      .loadOp = wgpu::LoadOp::Load,
      .storeOp = wgpu::StoreOp::Store,
  }};
  const wgpu::RenderPassDescriptor descriptor{
      .label = "Cloud shadows",
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes("Cloud shadows"),
  };
  const auto pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(g_cloudPass);
  pass.SetBindGroup(0, bindGroup, 0, nullptr);
  pass.Draw(3);
  pass.End();
}
} // namespace

void initialize() {
  const char* bloom = std::getenv("DOL_AURORA_BLOOM");
  g_bloom.store(bloom != nullptr && bloom[0] == '1', std::memory_order_relaxed);
  g_threshold = env_float("DOL_AURORA_BLOOM_THRESHOLD", g_threshold, 0.f, 1.f);
  g_strength = env_float("DOL_AURORA_BLOOM_STRENGTH", g_strength, 0.f, 4.f);
  const char* far = std::getenv("DOL_AURORA_FAR_BLUR");
  g_farBlur.store(far != nullptr && far[0] == '1', std::memory_order_relaxed);
  g_farStart = env_float("DOL_AURORA_FAR_BLUR_START", g_farStart, 1.f, 1e9f);
  g_farEnd = env_float("DOL_AURORA_FAR_BLUR_END", g_farEnd, g_farStart + 1.f, 1e9f);
  g_farAmount = env_float("DOL_AURORA_FAR_BLUR_AMOUNT", g_farAmount, 0.f, 1.f);
  g_farBackground = env_float("DOL_AURORA_FAR_BLUR_BACKGROUND", g_farBackground, 0.f, 1.f);
  const char* ao = std::getenv("DOL_AURORA_AO");
  g_ao.store(ao != nullptr && ao[0] == '1', std::memory_order_relaxed);
  g_aoRadius = env_float("DOL_AURORA_AO_RADIUS", g_aoRadius, 0.1f, 1e6f);
  g_aoStrength = env_float("DOL_AURORA_AO_STRENGTH", g_aoStrength, 0.f, 1.f);
  g_cloudStrength = env_float("DOL_AURORA_CLOUD_SHADOWS_STRENGTH", g_cloudStrength, 0.f, 1.f);
  g_cloudScale = env_float("DOL_AURORA_CLOUD_SHADOWS_SCALE", g_cloudScale, 1.f, 1e7f);
  const char* cloudDebug = std::getenv("DOL_AURORA_CLOUD_SHADOWS_DEBUG");
  g_cloudDebug = cloudDebug != nullptr && cloudDebug[0] == '1';
  const char* aoDebug = std::getenv("DOL_AURORA_AO_DEBUG");
  g_aoDebug = aoDebug != nullptr && aoDebug[0] == '1';
  const char* farDebug = std::getenv("DOL_AURORA_FAR_BLUR_DEBUG");
  g_farDebug = farDebug != nullptr && farDebug[0] == '1';

  wgpu::ShaderSourceWGSL source{};
  source.code = kShader;
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Post effects"};
  g_module = g_device.CreateShaderModule(&moduleDescriptor);
  const std::array layoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = {.type = wgpu::SamplerBindingType::Filtering},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "Post effects",
      .entryCount = layoutEntries.size(),
      .entries = layoutEntries.data(),
  };
  g_layout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{.bindGroupLayoutCount = 1, .bindGroupLayouts = &g_layout};
  g_pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  const wgpu::SamplerDescriptor samplerDescriptor{
      .label = "Post effects",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  g_sampler = g_device.CreateSampler(&samplerDescriptor);

  wgpu::ShaderSourceWGSL farSource{};
  farSource.code = kFarShader;
  const wgpu::ShaderModuleDescriptor farModuleDescriptor{.nextInChain = &farSource, .label = "Far blur"};
  g_farModule = g_device.CreateShaderModule(&farModuleDescriptor);
  wgpu::ShaderSourceWGSL aoSource{};
  aoSource.code = kAoShader;
  const wgpu::ShaderModuleDescriptor aoModuleDescriptor{.nextInChain = &aoSource, .label = "Ambient occlusion"};
  g_aoModule = g_device.CreateShaderModule(&aoModuleDescriptor);
  wgpu::ShaderSourceWGSL cloudSource{};
  cloudSource.code = kCloudShader;
  const wgpu::ShaderModuleDescriptor cloudModuleDescriptor{.nextInChain = &cloudSource, .label = "Cloud shadows"};
  g_cloudModule = g_device.CreateShaderModule(&cloudModuleDescriptor);
  const std::array depthEntries{
      layoutEntries[0],
      layoutEntries[1],
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Depth, .viewDimension = wgpu::TextureViewDimension::e2D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor depthLayoutDescriptor{
      .label = "Far blur",
      .entryCount = depthEntries.size(),
      .entries = depthEntries.data(),
  };
  g_depthLayout = g_device.CreateBindGroupLayout(&depthLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor depthPipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                     .bindGroupLayouts = &g_depthLayout};
  g_depthPipelineLayout = g_device.CreatePipelineLayout(&depthPipelineLayoutDescriptor);
  const std::array cloudEntries{
      depthEntries[0],
      depthEntries[1],
      depthEntries[2],
      wgpu::BindGroupLayoutEntry{
          .binding = 3,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(g_camera)},
      },
  };
  const wgpu::BindGroupLayoutDescriptor cloudLayoutDescriptor{
      .label = "Cloud shadows",
      .entryCount = cloudEntries.size(),
      .entries = cloudEntries.data(),
  };
  g_cloudLayout = g_device.CreateBindGroupLayout(&cloudLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor cloudPipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                     .bindGroupLayouts = &g_cloudLayout};
  g_cloudPipelineLayout = g_device.CreatePipelineLayout(&cloudPipelineLayoutDescriptor);
}

void shutdown() {
  g_bright = g_blurH = g_blurV = g_composite = g_down = g_farComposite = g_aoPass = g_aoApply = {};
  g_aoTex = {};
  g_aoView = {};
  g_aoSamples = 0;
  g_aoModule = {};
  g_cloudPass = {};
  g_cloudSamples = 0;
  g_cloudPipelineLayout = {};
  g_cloudLayout = {};
  g_cloudModule = {};
  g_tex = {};
  g_view = {};
  g_farTex = {};
  g_farView = {};
  g_farSamples = 0;
  g_depthPipelineLayout = {};
  g_depthLayout = {};
  g_farModule = {};
  g_size = {};
  g_format = wgpu::TextureFormat::Undefined;
  g_compositeSamples = 0;
  g_sampler = {};
  g_pipelineLayout = {};
  g_layout = {};
  g_module = {};
}

bool bloom_enabled() noexcept { return g_bloom.load(std::memory_order_relaxed); }

void set_bloom(bool enabled) noexcept { g_bloom.store(enabled, std::memory_order_relaxed); }

bool far_blur_enabled() noexcept { return g_farBlur.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_far_blur(bool enabled) noexcept { g_farBlur.store(enabled, std::memory_order_relaxed); }

bool ambient_occlusion_enabled() noexcept { return g_ao.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_ambient_occlusion(bool enabled) noexcept { g_ao.store(enabled, std::memory_order_relaxed); }

bool cloud_shadows_enabled() noexcept { return g_clouds.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_cloud_shadows(bool enabled) noexcept { g_clouds.store(enabled, std::memory_order_relaxed); }

void set_world_view(const float invView[12], float tanHalfFovy, float aspect, float nearPlane, float seaLevel,
                    float seconds) noexcept {
  std::lock_guard lock{g_cameraMutex};
  std::memcpy(g_camera.data(), invView, 12 * sizeof(float));
  g_camera[12] = tanHalfFovy;
  g_camera[13] = aspect;
  g_camera[14] = nearPlane;
  g_camera[15] = seaLevel;
  g_camera[16] = seconds;
  g_cameraValid = nearPlane > 0.f && tanHalfFovy > 0.f && aspect > 0.f;
}

bool any_enabled() noexcept {
  return bloom_enabled() || far_blur_enabled() || ambient_occlusion_enabled() || cloud_shadows_enabled();
}

void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
            const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
            const wgpu::TextureView& resolve, uint32_t msaaSamples) {
  // Ambient occlusion and the far blur need single-sample depth (the depth
  // buffer is multisampled with MSAA).
  if (ambient_occlusion_enabled() && g_module && depthView && msaaSamples == 1 && size.width >= 4 && size.height >= 4) {
    const webgpu::gpu_prof::Zone zone{cmd, "Ambient occlusion"};
    ensure_targets(size, format, msaaSamples);
    draw_depth(cmd, "Ambient occlusion", g_aoPass, g_aoView[1], depthView, g_aoView[0], nullptr, false);
    draw(cmd, "Ambient occlusion blur (horizontal)", g_blurH, g_aoView[0], g_aoView[1], false);
    draw(cmd, "Ambient occlusion blur (vertical)", g_blurV, g_aoView[1], g_aoView[0], false);
    // With DOL_AURORA_AO_DEBUG=1 the occlusion replaces the scene.
    draw(cmd, "Ambient occlusion apply", g_aoApply, g_aoView[0], dst, true, resolve);
  }
  if (cloud_shadows_enabled() && g_module && depthView && msaaSamples == 1) {
    std::array<float, 20> view;
    bool valid;
    {
      std::lock_guard lock{g_cameraMutex};
      view = g_camera;
      valid = g_cameraValid;
    }
    if (valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Cloud shadows"};
      ensure_targets(size, format, msaaSamples);
      draw_clouds(cmd, view, depthView, dst, resolve);
    }
  }
  if (far_blur_enabled() && g_module && depthView && msaaSamples == 1 && size.width >= 4 && size.height >= 4) {
    const webgpu::gpu_prof::Zone zone{cmd, "Far blur"};
    ensure_targets(size, format, msaaSamples);
    draw(cmd, "Far blur downsample", g_down, scene, g_farView[0], false);
    for (int i = 0; i < 2; ++i) {
      draw(cmd, "Far blur (horizontal)", g_blurH, g_farView[0], g_farView[1], false);
      draw(cmd, "Far blur (vertical)", g_blurV, g_farView[1], g_farView[0], false);
    }
    draw_depth(cmd, "Far blur composite", g_farComposite, g_farView[0], depthView, dst, resolve, true);
  }
  if (bloom_enabled())
    encode_bloom(cmd, scene, size, format, dst, resolve, msaaSamples);
}

void encode_bloom(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::Extent3D& size,
                  wgpu::TextureFormat format, const wgpu::TextureView& dst, const wgpu::TextureView& resolve,
                  uint32_t msaaSamples) {
  if (!g_module || !scene || !dst || size.width < 4 || size.height < 4) {
    return;
  }
  const webgpu::gpu_prof::Zone zone{cmd, "Bloom"};
  ensure_targets(size, format, msaaSamples);
  draw(cmd, "Bloom bright pass", g_bright, scene, g_view[0], false);
  for (int i = 0; i < 2; ++i) {
    draw(cmd, "Bloom blur (horizontal)", g_blurH, g_view[0], g_view[1], false);
    draw(cmd, "Bloom blur (vertical)", g_blurV, g_view[1], g_view[0], false);
  }
  draw(cmd, "Bloom composite", g_composite, g_view[0], dst, true, resolve);
}
} // namespace aurora::gfx::post_fx

void aurora_set_bloom(bool enabled) { aurora::gfx::post_fx::set_bloom(enabled); }

void aurora_set_far_blur(bool enabled) { aurora::gfx::post_fx::set_far_blur(enabled); }

void aurora_set_ambient_occlusion(bool enabled) { aurora::gfx::post_fx::set_ambient_occlusion(enabled); }

void aurora_set_cloud_shadows(bool enabled) { aurora::gfx::post_fx::set_cloud_shadows(enabled); }

void aurora_set_world_view(const float inv_view[12], float tan_half_fovy, float aspect, float near_plane,
                           float sea_level, float seconds) {
  aurora::gfx::post_fx::set_world_view(inv_view, tan_half_fovy, aspect, near_plane, sea_level, seconds);
}
