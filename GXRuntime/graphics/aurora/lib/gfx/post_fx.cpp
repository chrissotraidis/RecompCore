#include "post_fx.hpp"

#include <aurora/aurora.h>

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <algorithm>
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
std::atomic_bool g_sun{false};
std::atomic_bool g_edge{false};
std::atomic_bool g_torch{false};
float g_torchStrength = 0.35f;  // how bright a light's glow is
float g_torchRadiusScale = 3.f; // the glow's reach, times the radius the game lights with
// The game's point lights (aurora_set_point_lights): x, y, z, radius, r, g, b, flicker each.
std::array<float, kMaxLights * 8> g_lights{};
uint32_t g_lightCount = 0;
float g_edgeStrength = 0.4f;  // how bright a sunlit edge gets
float g_sunStrength = 0.35f;  // how dark a sun shadow is
float g_sunReach = 1500.f;    // how far toward the sun a shadow is looked for, in world units
std::atomic_bool g_sunMapOn{false};
uint32_t g_sunMapSize = 2048;    // the shadow map's texels a side
float g_sunMapRange = 5000.f;    // world units from its centre to its edge
float g_sunMapDepth = 20000.f;   // world units toward and away from the sun it holds, from its centre
float g_sunMapBias = 12.f;       // world units a surface is moved toward the sun before the lookup
float g_sunMapContact = 300.f;   // the screen's steps' reach, for what the map leaves out
bool g_sunMapDebug = false;      // DOL_AURORA_SUN_SHADOW_MAP_DEBUG=1: the map's shadow alone, gray outside it
float g_cloudStrength = 0.28f;  // how dark a cloud's shadow is
float g_cloudScale = 7000.f;    // world units across one cloud
bool g_cloudDebug = false;
// The camera, for the cloud shadows (aurora_set_world_view): the inverse view
// matrix's three rows, then tan(fovy / 2), aspect, near, the sea's height, and
// the time in seconds, then the direction to the sun (aurora_set_sun_direction).
// Written by the game thread, read when a frame is encoded.
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

// Half an 8-bit step of noise (interleaved gradient noise): a smooth shade
// multiplied or added into the 8-bit scene otherwise shows as bands.
fn dither(pos: vec2<f32>) -> f32 {
    return (fract(52.9829189 * fract(dot(pos, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
}

// Multiplied into the scene (ambient occlusion, sun shadows); the EFB's alpha is kept.
@fragment
fn fs_ao_apply(in: VertexOutput) -> @location(0) vec4<f32> {
    let shade = textureSample(src, samp, in.uv).rgb;
    return vec4(shade + select(0.0, dither(in.pos.xy), shade.g < 0.999), 1.0);
}

// Added onto the scene; the EFB's alpha is kept.
@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let glow = textureSample(src, samp, in.uv).rgb * strength;
    return vec4(max(glow + select(0.0, dither(in.pos.xy), max(glow.r, max(glow.g, glow.b)) > 0.002), vec3(0.0)), 0.0);
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

// A gradient for each lattice point, from its integer coordinates: no sin(), so
// every GPU gets the same values whatever its precision.
fn gradient(p: vec2<f32>) -> vec2<f32> {
    var q = bitcast<vec2<u32>>(vec2<i32>(p));
    q = q * 1664525u + 1013904223u;
    q.x += q.y * 1664525u;
    q.y += q.x * 1664525u;
    q = q ^ (q >> vec2(16u));
    q.x += q.y * 1664525u;
    q.y += q.x * 1664525u;
    q = q ^ (q >> vec2(16u));
    let a = f32(q.x) * (6.2831853 / 4294967296.0);
    return vec2(cos(a), sin(a));
}

// Gradient noise, about 0 to 1. Value noise put the clouds' edges along the
// world's X and Z (square-looking shadows); gradient noise has no such rows.
fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = dot(gradient(i), f);
    let b = dot(gradient(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    let c = dot(gradient(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    let e = dot(gradient(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));
    return 0.5 + 0.9 * mix(mix(a, b, u.x), mix(c, e, u.x), u.y);
}

fn fbm(p0: vec2<f32>) -> f32 {
    // Each octave turned by about 37 degrees, so no two line up.
    let turn = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    var p = p0;
    var v = 0.0;
    var a = 0.5;
    for (var i = 0; i < 4; i++) {
        v += a * gradient_noise(p);
        p = turn * p * 2.03 + vec2(17.0, 9.0);
        a *= 0.5;
    }
    return v;
}

// Half an 8-bit step of noise (interleaved gradient noise), so a smooth shade
// multiplied into the 8-bit scene doesn't come out as bands on the sea.
fn dither(pos: vec2<f32>) -> f32 {
    return (fract(52.9829189 * fract(dot(pos, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
}

@fragment
fn fs_clouds(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureLoad(depth, vec2<i32>(in.pos.xy), 0);
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let ray = vec3(ndc.x * view.lens.x * view.lens.y, ndc.y * view.lens.x, -1.0);
    let dir = vec3(dot(view.rows[0].xyz, ray), dot(view.rows[1].xyz, ray), dot(view.rows[2].xyz, ray));
    let eye = vec3(view.rows[0].w, view.rows[1].w, view.rows[2].w);
    // Where the view ray meets the sea, if it does.
    var sea = -1.0;
    if (dir.y < -1e-4) {
        sea = (view.lens.w - eye.y) / dir.y;
    }
    var world = eye;
    if (d > 0.0 && (sea <= 0.0 || view.lens.z / d < 0.9 * sea)) {
        // Something in front of the sea (an island, a boat): its place from depth.
        world = eye + dir * (view.lens.z / d);
    } else {
        // The sea, also where it has no depth: its place from its plane. Taking
        // the sea's distance from depth too gave straight-edged steps far out
        // wherever depth is less precise (seen on Vulkan).
        if (sea <= 0.0) {
            return vec4(1.0);
        }
        world = eye + dir * sea;
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
    return vec4(vec3(1.0 - shade + select(0.0, dither(in.pos.xy), shade > 0.0)), 1.0);
}
)""";
wgpu::ShaderModule g_cloudModule;

// Sun shadows from what is on screen: from each pixel's place in view space,
// steps toward the sun; a step that lands behind the depth buffer's surface (by
// less than an object's assumed thickness) is in shadow. Off-screen casters
// cast nothing; the sun low or below the horizon casts nothing. Drawn at a
// quarter of the size and blurred (the jittered steps' grain becomes a soft
// edge), then multiplied into the scene.
constexpr const char* kSunShader = R"""(
override strength: f32 = 0.35;
override reach: f32 = 1500.0;

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds, then the direction to the sun (world)
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

fn view_depth(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth));
    let p = clamp(vec2<i32>(uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let d = textureLoad(depth, p, 0);
    return select(view.lens.z / d, 0.0, d <= 0.0);
}

@fragment
fn fs_sun(in: VertexOutput) -> @location(0) vec4<f32> {
    let z = view_depth(in.uv);
    let sun = view.time.yzw;
    if (z <= 0.0 || sun.y < 0.05) {
        return vec4(1.0);
    }
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let tx = view.lens.x * view.lens.y;
    let ty = view.lens.x;
    let p = vec3(ndc.x * tx, ndc.y * ty, -1.0) * z;
    // The sun in view space: the inverse view's rotation transposed.
    let l = normalize(vec3(dot(vec3(view.rows[0].x, view.rows[1].x, view.rows[2].x), sun),
                           dot(vec3(view.rows[0].y, view.rows[1].y, view.rows[2].y), sun),
                           dot(vec3(view.rows[0].z, view.rows[1].z, view.rows[2].z), sun)));
    let jitter = fract(sin(dot(in.pos.xy, vec2(12.9898, 78.233))) * 43758.5453);
    var lit = 1.0;
    for (var i = 0; i < 24; i++) {
        let t = 20.0 + (reach - 20.0) * (f32(i) + jitter) / 24.0;
        let q = p + l * t;
        if (-q.z <= view.lens.z) {
            break;
        }
        let s = vec2(q.x / -q.z / tx, q.y / -q.z / ty);
        let uv = vec2(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        let surface = view_depth(uv);
        let behind = -q.z - surface;
        if (surface > 0.0 && behind > 12.0 + 0.004 * surface && behind < 250.0) {
            lit = 0.0;
            break;
        }
    }
    let fade = (1.0 - smoothstep(4000.0, 9000.0, z)) * smoothstep(0.05, 0.25, sun.y);
    return vec4(vec3(1.0 - strength * (1.0 - lit) * fade), 1.0);
}
)""";
wgpu::ShaderModule g_sunModule;

// Sun shadows from the shadow map: each pixel's place in the world, taken to
// the map (view.light), is in shadow where something nearer the sun was drawn
// there (nine filtered lookups around it, for a soft edge). The map fades out
// toward its edge. Alpha-tested draws are not in the map (leaves, fences), so
// a few of the screen's steps (as fs_sun, out to `contact`) still look for
// them. Drawn at a quarter of the size, blurred and multiplied in, as fs_sun.
constexpr const char* kSunMapShader = R"""(
override strength: f32 = 0.35;
override contact: f32 = 300.0;
override debug_view: f32 = 0.0;

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds, then the direction to the sun (world)
    light: array<vec4<f32>, 3>, // the world to the map (post_fx::SunMap::world)
    params: vec4<f32>,          // a texel's size in the map, the depth bias
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var unused: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var sun_map: texture_depth_2d;
@group(0) @binding(5) var sun_compare: sampler_comparison;

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

fn view_depth(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth));
    let p = clamp(vec2<i32>(uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let d = textureLoad(depth, p, 0);
    return select(view.lens.z / d, 0.0, d <= 0.0);
}

@fragment
fn fs_sun_map(in: VertexOutput) -> @location(0) vec4<f32> {
    let z = view_depth(in.uv);
    let sun = view.time.yzw;
    if (z <= 0.0 || sun.y < 0.05) {
        return vec4(1.0);
    }
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let tx = view.lens.x * view.lens.y;
    let ty = view.lens.x;
    let p = vec3(ndc.x * tx, ndc.y * ty, -1.0) * z;
    let world = vec4(dot(view.rows[0], vec4(p, 1.0)), dot(view.rows[1], vec4(p, 1.0)), dot(view.rows[2], vec4(p, 1.0)), 1.0);
    let m = vec3(dot(view.light[0], world), dot(view.light[1], world), dot(view.light[2], world));
    var lit = 1.0;
    let edge = max(abs(m.x), abs(m.y));
    if (edge < 1.0 && m.z > 0.0 && m.z < 1.0) {
        let uv = vec2(m.x * 0.5 + 0.5, 0.5 - m.y * 0.5);
        let reference = m.z - view.params.y;
        var sum = 0.0;
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                sum += textureSampleCompareLevel(sun_map, sun_compare, uv + vec2(f32(i), f32(j)) * view.params.x, reference);
            }
        }
        lit = mix(sum / 9.0, 1.0, smoothstep(0.85, 1.0, edge));
    } else if (debug_view > 0.5) {
        return vec4(vec3(0.5), 1.0);
    }
    if (debug_view > 0.5) {
        return vec4(vec3(lit), 1.0);
    }
    // The screen's steps, a short way (fs_sun).
    let l = normalize(vec3(dot(vec3(view.rows[0].x, view.rows[1].x, view.rows[2].x), sun),
                           dot(vec3(view.rows[0].y, view.rows[1].y, view.rows[2].y), sun),
                           dot(vec3(view.rows[0].z, view.rows[1].z, view.rows[2].z), sun)));
    let jitter = fract(sin(dot(in.pos.xy, vec2(12.9898, 78.233))) * 43758.5453);
    for (var i = 0; i < 8; i++) {
        let t = 10.0 + (contact - 10.0) * (f32(i) + jitter) / 8.0;
        let q = p + l * t;
        if (-q.z <= view.lens.z) {
            break;
        }
        let s = vec2(q.x / -q.z / tx, q.y / -q.z / ty);
        let suv = vec2(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
            break;
        }
        let surface = view_depth(suv);
        let behind = -q.z - surface;
        if (surface > 0.0 && behind > 12.0 + 0.004 * surface && behind < 250.0) {
            lit = min(lit, smoothstep(1500.0, 4000.0, z));  // near the camera only
            break;
        }
    }
    let fade = smoothstep(0.05, 0.25, sun.y);
    return vec4(vec3(1.0 - strength * (1.0 - lit) * fade), 1.0);
}
)""";
wgpu::ShaderModule g_sunMapModule;
wgpu::BindGroupLayout g_sunMapLayout;
wgpu::PipelineLayout g_sunMapPipelineLayout;
wgpu::RenderPipeline g_sunMapPass;
wgpu::Sampler g_sunMapCompare;
wgpu::Texture g_sunMapTex;
wgpu::TextureView g_sunMapView;

// Edge light: a warm rim on the sunlit outline of what is near the camera (Link,
// people, objects). A pixel is on such an outline when, a few pixels from it
// toward the sun on screen, the depth jumps far behind it or there is none.
constexpr const char* kEdgeShader = R"""(
override strength: f32 = 0.4;

struct View {
    rows: array<vec4<f32>, 3>,
    lens: vec4<f32>,
    time: vec4<f32>,
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

fn view_depth(p: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth));
    let d = textureLoad(depth, clamp(p, vec2<i32>(0), size - vec2<i32>(1)), 0);
    return select(view.lens.z / d, 0.0, d <= 0.0);
}

@fragment
fn fs_edge(in: VertexOutput) -> @location(0) vec4<f32> {
    let here = vec2<i32>(in.pos.xy);
    let z = view_depth(here);
    let sun = view.time.yzw;
    if (z <= 0.0 || z > 1500.0 || sun.y < 0.05) {
        return vec4(0.0);
    }
    // The sun's direction in view space, then on screen (y down).
    let l = vec3(dot(vec3(view.rows[0].x, view.rows[1].x, view.rows[2].x), sun),
                 dot(vec3(view.rows[0].y, view.rows[1].y, view.rows[2].y), sun),
                 dot(vec3(view.rows[0].z, view.rows[1].z, view.rows[2].z), sun));
    var dir = vec2(l.x, -l.y);
    if (length(dir) < 0.05) {
        dir = vec2(0.0, -1.0);
    }
    dir = normalize(dir);
    let scale = f32(textureDimensions(depth).y) / 1080.0;
    var rim = 0.0;
    for (var i = 1; i <= 3; i++) {
        let n = view_depth(here + vec2<i32>(round(dir * f32(i) * 1.5 * scale)));
        if (n <= 0.0 || n > z * 1.5 + 100.0) {
            rim = max(rim, 1.0 - f32(i - 1) / 3.0);
        }
    }
    let fade = 1.0 - smoothstep(800.0, 1500.0, z);
    return vec4(vec3(1.0, 0.95, 0.82) * rim * strength * fade, 0.0);
}
)""";
wgpu::ShaderModule g_edgeModule;

// Torch and fire light: the game's own point lights (torches, fires, lanterns,
// bombs) glow on the surfaces around them, each pixel's place in the world from
// the camera and the depth, with a smooth falloff to the light's radius.
constexpr const char* kTorchShader = R"""(
override strength: f32 = 0.35;
override radius_scale: f32 = 3.0;

struct Light {
    place: vec4<f32>,  // x, y, z, radius
    color: vec4<f32>,  // r, g, b (0 to 1), flicker
};

struct View {
    rows: array<vec4<f32>, 3>,
    lens: vec4<f32>,
    time: vec4<f32>,
    count: vec4<f32>,
    lights: array<Light, 16>,
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

@fragment
fn fs_torch(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureLoad(depth, vec2<i32>(in.pos.xy), 0);
    if (d <= 0.0) {
        return vec4(0.0);
    }
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let ray = vec3(ndc.x * view.lens.x * view.lens.y, ndc.y * view.lens.x, -1.0);
    let dir = vec3(dot(view.rows[0].xyz, ray), dot(view.rows[1].xyz, ray), dot(view.rows[2].xyz, ray));
    let eye = vec3(view.rows[0].w, view.rows[1].w, view.rows[2].w);
    let world = eye + dir * (view.lens.z / d);
    var glow = vec3(0.0);
    let n = u32(view.count.x);
    for (var i = 0u; i < 16u; i++) {
        if (i >= n) {
            break;
        }
        let light = view.lights[i];
        let falloff = clamp(1.0 - length(world - light.place.xyz) / (light.place.w * radius_scale), 0.0, 1.0);
        let flicker = 1.0 + light.color.w * 0.15 * sin(view.time.x * 9.0 + f32(i) * 1.7);
        glow += light.color.rgb * falloff * falloff * flicker;
    }
    return vec4(glow * strength, 0.0);
}
)""";
wgpu::ShaderModule g_torchModule;
wgpu::BindGroupLayout g_torchLayout;
wgpu::PipelineLayout g_torchPipelineLayout;
wgpu::RenderPipeline g_torchPass;
uint32_t g_torchSamples = 0;
constexpr size_t kTorchUniformSize = 20 * sizeof(float) + 4 * sizeof(float) + kMaxLights * 8 * sizeof(float);
wgpu::RenderPipeline g_edgePass;
uint32_t g_edgeSamples = 0;
wgpu::RenderPipeline g_sunPass;
uint32_t g_sunSamples = 0;
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

wgpu::RenderPipeline make_sun_pipeline(wgpu::TextureFormat format) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_sunStrength},
      wgpu::ConstantEntry{.key = "reach", .value = g_sunReach},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = nullptr, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_sunModule,
      .entryPoint = "fs_sun",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Sun shadows",
      .layout = g_cloudPipelineLayout,
      .vertex = {.module = g_sunModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_sun_map_pipeline(wgpu::TextureFormat format) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_sunStrength},
      wgpu::ConstantEntry{.key = "contact", .value = g_sunMapContact},
      wgpu::ConstantEntry{.key = "debug_view", .value = g_sunMapDebug ? 1.0 : 0.0},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = nullptr, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_sunMapModule,
      .entryPoint = "fs_sun_map",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Sun shadows (shadow map)",
      .layout = g_sunMapPipelineLayout,
      .vertex = {.module = g_sunMapModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_edge_pipeline(wgpu::TextureFormat format, uint32_t samples) {
  const std::array constants{wgpu::ConstantEntry{.key = "strength", .value = g_edgeStrength}};
  const wgpu::BlendState add{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = &add, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_edgeModule,
      .entryPoint = "fs_edge",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Edge light",
      .layout = g_cloudPipelineLayout,
      .vertex = {.module = g_edgeModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_torch_pipeline(wgpu::TextureFormat format, uint32_t samples) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_torchStrength},
      wgpu::ConstantEntry{.key = "radius_scale", .value = g_torchRadiusScale},
  };
  const wgpu::BlendState add{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = &add, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_torchModule,
      .entryPoint = "fs_torch",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Torch light",
      .layout = g_torchPipelineLayout,
      .vertex = {.module = g_torchModule, .entryPoint = "vs_main"},
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
    g_sunPass = {};
    g_sunMapPass = {};
    g_edgePass = {};
    g_torchPass = {};
    g_composite = {};
    g_farComposite = {};
    g_aoApply = {};
    g_size = {};
  }
  if (!g_composite || samples != g_compositeSamples) {
    g_composite = make_pipeline("Bloom composite", "fs_composite", format, samples, kAdd);
    g_compositeSamples = samples;
  }
  if (!g_sunPass) {
    g_sunPass = make_sun_pipeline(format);
  }
  if (!g_sunMapPass) {
    g_sunMapPass = make_sun_map_pipeline(format);
  }
  if (!g_torchPass || samples != g_torchSamples) {
    g_torchPass = make_torch_pipeline(format, samples);
    g_torchSamples = samples;
  }
  if (!g_edgePass || samples != g_edgeSamples) {
    g_edgePass = make_edge_pipeline(format, samples);
    g_edgeSamples = samples;
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
void draw_with_uniform(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                       const wgpu::BindGroupLayout& layout, const void* uniform, size_t uniformSize,
                       const wgpu::TextureView& depthView, const wgpu::TextureView& target,
                       const wgpu::TextureView& resolve);

void draw_with_view(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                    const std::array<float, 20>& view, const wgpu::TextureView& depthView,
                    const wgpu::TextureView& target, const wgpu::TextureView& resolve) {
  draw_with_uniform(cmd, label, pipeline, g_cloudLayout, view.data(), sizeof(view), depthView, target, resolve);
}

void draw_with_uniform(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                       const wgpu::BindGroupLayout& layout, const void* uniform, size_t uniformSize,
                       const wgpu::TextureView& depthView, const wgpu::TextureView& target,
                       const wgpu::TextureView& resolve) {
  // A small buffer for each frame, so a frame still in flight keeps its camera.
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Post effects view",
      .usage = wgpu::BufferUsage::Uniform,
      .size = uniformSize,
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, uniformSize), uniform, uniformSize);
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = g_farView[0]},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .buffer = buffer, .offset = 0, .size = uniformSize},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = layout,
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
// The sun shadows from the shadow map (kSunMapShader) into `target`.
void draw_sun_map(const wgpu::CommandEncoder& cmd, const Camera& camera, const SunMap& sunMap,
                  const wgpu::TextureView& depthView, const wgpu::TextureView& target) {
  std::array<float, 36> uniform{};
  std::memcpy(uniform.data(), camera.view.data(), sizeof(camera.view));
  std::memcpy(uniform.data() + 20, sunMap.world.data(), sizeof(sunMap.world));
  uniform[32] = 1.f / static_cast<float>(g_sunMapSize);
  uniform[33] = g_sunMapBias / (2.f * g_sunMapDepth);
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Sun shadow map view",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(uniform),
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, sizeof(uniform)), uniform.data(), sizeof(uniform));
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = g_farView[0]},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .buffer = buffer, .offset = 0, .size = sizeof(uniform)},
      wgpu::BindGroupEntry{.binding = 4, .textureView = g_sunMapView},
      wgpu::BindGroupEntry{.binding = 5, .sampler = g_sunMapCompare},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_sunMapLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  const std::array attachments{wgpu::RenderPassColorAttachment{
      .view = target,
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {1.0, 1.0, 1.0, 1.0},
  }};
  const wgpu::RenderPassDescriptor descriptor{
      .label = "Sun shadows (shadow map)",
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes("Sun shadows (shadow map)"),
  };
  const auto pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(g_sunMapPass);
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
  g_sunStrength = env_float("DOL_AURORA_SUN_SHADOWS_STRENGTH", g_sunStrength, 0.f, 1.f);
  g_sunReach = env_float("DOL_AURORA_SUN_SHADOWS_REACH", g_sunReach, 1.f, 1e6f);
  const char* sunMap = std::getenv("DOL_AURORA_SUN_SHADOW_MAP");
  g_sunMapOn.store(sunMap != nullptr && sunMap[0] == '1', std::memory_order_relaxed);
  g_sunMapSize = static_cast<uint32_t>(env_float("DOL_AURORA_SUN_SHADOW_MAP_SIZE", static_cast<float>(g_sunMapSize), 256.f, 8192.f));
  g_sunMapRange = env_float("DOL_AURORA_SUN_SHADOW_MAP_RANGE", g_sunMapRange, 500.f, 100000.f);
  g_sunMapDepth = env_float("DOL_AURORA_SUN_SHADOW_MAP_DEPTH", g_sunMapDepth, 1000.f, 1000000.f);
  g_sunMapBias = env_float("DOL_AURORA_SUN_SHADOW_MAP_BIAS", g_sunMapBias, 0.f, 1000.f);
  g_sunMapContact = env_float("DOL_AURORA_SUN_SHADOW_MAP_CONTACT", g_sunMapContact, 20.f, 5000.f);
  const char* sunMapDebug = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_DEBUG");
  g_sunMapDebug = sunMapDebug != nullptr && sunMapDebug[0] == '1';
  g_edgeStrength = env_float("DOL_AURORA_EDGE_LIGHT_STRENGTH", g_edgeStrength, 0.f, 4.f);
  g_torchStrength = env_float("DOL_AURORA_TORCH_LIGHT_STRENGTH", g_torchStrength, 0.f, 4.f);
  g_torchRadiusScale = env_float("DOL_AURORA_TORCH_LIGHT_REACH", g_torchRadiusScale, 0.1f, 50.f);
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
  wgpu::ShaderSourceWGSL sunSource{};
  sunSource.code = kSunShader;
  const wgpu::ShaderModuleDescriptor sunModuleDescriptor{.nextInChain = &sunSource, .label = "Sun shadows"};
  g_sunModule = g_device.CreateShaderModule(&sunModuleDescriptor);
  wgpu::ShaderSourceWGSL edgeSource{};
  edgeSource.code = kEdgeShader;
  const wgpu::ShaderModuleDescriptor edgeModuleDescriptor{.nextInChain = &edgeSource, .label = "Edge light"};
  g_edgeModule = g_device.CreateShaderModule(&edgeModuleDescriptor);
  wgpu::ShaderSourceWGSL torchSource{};
  torchSource.code = kTorchShader;
  const wgpu::ShaderModuleDescriptor torchModuleDescriptor{.nextInChain = &torchSource, .label = "Torch light"};
  g_torchModule = g_device.CreateShaderModule(&torchModuleDescriptor);
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
  auto torchEntries = cloudEntries;
  torchEntries[3].buffer.minBindingSize = kTorchUniformSize;
  const wgpu::BindGroupLayoutDescriptor torchLayoutDescriptor{
      .label = "Torch light",
      .entryCount = torchEntries.size(),
      .entries = torchEntries.data(),
  };
  g_torchLayout = g_device.CreateBindGroupLayout(&torchLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor torchPipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                     .bindGroupLayouts = &g_torchLayout};
  g_torchPipelineLayout = g_device.CreatePipelineLayout(&torchPipelineLayoutDescriptor);

  wgpu::ShaderSourceWGSL sunMapSource{};
  sunMapSource.code = kSunMapShader;
  const wgpu::ShaderModuleDescriptor sunMapModuleDescriptor{.nextInChain = &sunMapSource,
                                                            .label = "Sun shadows (shadow map)"};
  g_sunMapModule = g_device.CreateShaderModule(&sunMapModuleDescriptor);
  std::array<wgpu::BindGroupLayoutEntry, 6> sunMapEntries{
      cloudEntries[0],
      cloudEntries[1],
      cloudEntries[2],
      cloudEntries[3],
      wgpu::BindGroupLayoutEntry{
          .binding = 4,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Depth, .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 5,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = {.type = wgpu::SamplerBindingType::Comparison},
      },
  };
  sunMapEntries[3].buffer.minBindingSize = 36 * sizeof(float);
  const wgpu::BindGroupLayoutDescriptor sunMapLayoutDescriptor{
      .label = "Sun shadows (shadow map)",
      .entryCount = sunMapEntries.size(),
      .entries = sunMapEntries.data(),
  };
  g_sunMapLayout = g_device.CreateBindGroupLayout(&sunMapLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor sunMapPipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                      .bindGroupLayouts = &g_sunMapLayout};
  g_sunMapPipelineLayout = g_device.CreatePipelineLayout(&sunMapPipelineLayoutDescriptor);
  const wgpu::SamplerDescriptor compareDescriptor{
      .label = "Sun shadow map",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .compare = wgpu::CompareFunction::LessEqual,
  };
  g_sunMapCompare = g_device.CreateSampler(&compareDescriptor);
}

void shutdown() {
  g_bright = g_blurH = g_blurV = g_composite = g_down = g_farComposite = g_aoPass = g_aoApply = {};
  g_aoTex = {};
  g_aoView = {};
  g_aoSamples = 0;
  g_aoModule = {};
  g_cloudPass = {};
  g_cloudSamples = 0;
  g_sunPass = {};
  g_sunSamples = 0;
  g_sunModule = {};
  g_sunMapPass = {};
  g_sunMapModule = {};
  g_sunMapPipelineLayout = {};
  g_sunMapLayout = {};
  g_sunMapCompare = {};
  g_sunMapView = {};
  g_sunMapTex = {};
  g_edgePass = {};
  g_edgeSamples = 0;
  g_edgeModule = {};
  g_torchPass = {};
  g_torchSamples = 0;
  g_torchPipelineLayout = {};
  g_torchLayout = {};
  g_torchModule = {};
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

bool sun_shadows_enabled() noexcept { return g_sun.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_sun_shadows(bool enabled) noexcept { g_sun.store(enabled, std::memory_order_relaxed); }

bool sun_shadow_map_enabled() noexcept { return sun_shadows_enabled() && g_sunMapOn.load(std::memory_order_relaxed); }

void set_sun_shadow_map(bool enabled) noexcept { g_sunMapOn.store(enabled, std::memory_order_relaxed); }

SunMap sun_map(const Camera& camera) noexcept {
  SunMap map;
  const float* v = camera.view.data();
  const float sun[3] = {v[17], v[18], v[19]};
  if (!camera.valid || sun[1] < 0.05f)
    return map;
  // Across the map: perpendicular to the sun, the first level.
  const float up[3] = {0.f, std::fabs(sun[1]) > 0.99f ? 0.f : 1.f, std::fabs(sun[1]) > 0.99f ? 1.f : 0.f};
  float x[3] = {up[1] * sun[2] - up[2] * sun[1], up[2] * sun[0] - up[0] * sun[2], up[0] * sun[1] - up[1] * sun[0]};
  const float xl = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  for (float& c : x)
    c /= xl;
  const float y[3] = {sun[1] * x[2] - sun[2] * x[1], sun[2] * x[0] - sun[0] * x[2], sun[0] * x[1] - sun[1] * x[0]};
  // The centre: half the range ahead of the camera, level (the view looks down
  // its -z, the inverse view's third column in the world).
  float ahead[3] = {-v[2], 0.f, -v[10]};
  const float al = std::sqrt(ahead[0] * ahead[0] + ahead[2] * ahead[2]);
  const float centre[3] = {v[3] + (al > 1e-4f ? ahead[0] / al : 0.f) * g_sunMapRange * 0.5f, v[7],
                           v[11] + (al > 1e-4f ? ahead[2] / al : 0.f) * g_sunMapRange * 0.5f};
  // Moved by whole texels across the map, so its texels stay where they are on
  // the ground while the camera moves and the shadows' edges don't crawl.
  const float texel = 2.f * g_sunMapRange / static_cast<float>(g_sunMapSize);
  const float cx = std::floor((x[0] * centre[0] + x[1] * centre[1] + x[2] * centre[2]) / texel) * texel;
  const float cy = std::floor((y[0] * centre[0] + y[1] * centre[1] + y[2] * centre[2]) / texel) * texel;
  const float cz = sun[0] * centre[0] + sun[1] * centre[1] + sun[2] * centre[2];
  const float r = 1.f / g_sunMapRange, d = 1.f / (2.f * g_sunMapDepth);
  map.world = {x[0] * r,    x[1] * r,    x[2] * r,    -cx * r,  //
               y[0] * r,    y[1] * r,    y[2] * r,    -cy * r,  //
               -sun[0] * d, -sun[1] * d, -sun[2] * d, 0.5f + cz * d};
  // The camera's view space there: world = A view + t, the inverse view.
  for (int row = 0; row < 3; ++row) {
    const float* w = &map.world[row * 4];
    for (int c = 0; c < 4; ++c) {
      float sum = c == 3 ? w[3] : 0.f;
      for (int k = 0; k < 3; ++k)
        sum += w[k] * v[k * 4 + c];
      map.caster[row * 4 + c] = sum;
    }
  }
  map.valid = true;
  return map;
}

const wgpu::TextureView& sun_map_view() {
  if (!g_sunMapTex || g_sunMapTex.GetWidth() != g_sunMapSize) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Sun shadow map",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {g_sunMapSize, g_sunMapSize, 1},
        .format = kSunMapFormat,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    g_sunMapTex = g_device.CreateTexture(&descriptor);
    g_sunMapView = g_sunMapTex.CreateView();
  }
  return g_sunMapView;
}

bool torch_light_enabled() noexcept { return g_torch.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_torch_light(bool enabled) noexcept { g_torch.store(enabled, std::memory_order_relaxed); }

void set_point_lights(const float* lights, uint32_t count) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_lightCount = std::min(count, kMaxLights);
  if (g_lightCount > 0 && lights != nullptr)
    std::memcpy(g_lights.data(), lights, g_lightCount * 8 * sizeof(float));
}

bool edge_light_enabled() noexcept { return g_edge.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_edge_light(bool enabled) noexcept { g_edge.store(enabled, std::memory_order_relaxed); }

void set_sun_direction(float x, float y, float z) noexcept {
  const float length = std::sqrt(x * x + y * y + z * z);
  std::lock_guard lock{g_cameraMutex};
  if (length > 0.f) {
    g_camera[17] = x / length;
    g_camera[18] = y / length;
    g_camera[19] = z / length;
  } else {
    g_camera[17] = g_camera[18] = g_camera[19] = 0.f;
  }
}

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

Camera current_camera() noexcept {
  Camera camera;
  std::lock_guard lock{g_cameraMutex};
  camera.view = g_camera;
  camera.valid = g_cameraValid;
  camera.lights = g_lights;
  camera.lightCount = g_lightCount;
  return camera;
}

Camera moved_camera(const Camera& camera, const float relative[3][4], float seconds) noexcept {
  Camera moved = camera;
  for (int r = 0; r < 3; ++r) {
    const float* row = &camera.view[r * 4];
    for (int c = 0; c < 4; ++c)
      moved.view[r * 4 + c] =
          row[0] * relative[0][c] + row[1] * relative[1][c] + row[2] * relative[2][c] + (c == 3 ? row[3] : 0.f);
  }
  moved.view[16] = seconds;
  return moved;
}

bool any_enabled() noexcept {
  return bloom_enabled() || far_blur_enabled() || ambient_occlusion_enabled() || cloud_shadows_enabled() ||
         sun_shadows_enabled() || edge_light_enabled() || torch_light_enabled();
}

void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
            const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
            const wgpu::TextureView& resolve, uint32_t msaaSamples, const Camera& camera, const SunMap* sunMap) {
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
  if (sun_shadows_enabled() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Sun shadows"};
      ensure_targets(size, format, msaaSamples);
      if (sunMap != nullptr && sunMap->valid && g_sunMapView)
        draw_sun_map(cmd, camera, *sunMap, depthView, g_aoView[0]);
      else
        draw_with_view(cmd, "Sun shadows", g_sunPass, camera.view, depthView, g_aoView[0], nullptr);
      draw(cmd, "Sun shadows blur (horizontal)", g_blurH, g_aoView[0], g_aoView[1], false);
      draw(cmd, "Sun shadows blur (vertical)", g_blurV, g_aoView[1], g_aoView[0], false);
      draw(cmd, "Sun shadows apply", g_aoApply, g_aoView[0], dst, true, resolve);
    }
  }
  if (edge_light_enabled() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Edge light"};
      ensure_targets(size, format, msaaSamples);
      draw_with_view(cmd, "Edge light", g_edgePass, camera.view, depthView, dst, resolve);
    }
  }
  if (torch_light_enabled() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid && camera.lightCount > 0) {
      std::array<float, kTorchUniformSize / sizeof(float)> uniform{};
      std::memcpy(uniform.data(), camera.view.data(), sizeof(camera.view));
      uniform[20] = static_cast<float>(camera.lightCount);
      std::memcpy(uniform.data() + 24, camera.lights.data(), sizeof(camera.lights));
      const webgpu::gpu_prof::Zone zone{cmd, "Torch light"};
      ensure_targets(size, format, msaaSamples);
      draw_with_uniform(cmd, "Torch light", g_torchPass, g_torchLayout, uniform.data(), sizeof(uniform), depthView,
                        dst, resolve);
    }
  }
  if (cloud_shadows_enabled() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Cloud shadows"};
      ensure_targets(size, format, msaaSamples);
      draw_with_view(cmd, "Cloud shadows", g_cloudPass, camera.view, depthView, dst, resolve);
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

void aurora_set_sun_shadows(bool enabled) { aurora::gfx::post_fx::set_sun_shadows(enabled); }

void aurora_set_sun_direction(float x, float y, float z) { aurora::gfx::post_fx::set_sun_direction(x, y, z); }

void aurora_set_sun_shadow_map(bool enabled) { aurora::gfx::post_fx::set_sun_shadow_map(enabled); }

void aurora_set_edge_light(bool enabled) { aurora::gfx::post_fx::set_edge_light(enabled); }

void aurora_set_torch_light(bool enabled) { aurora::gfx::post_fx::set_torch_light(enabled); }

void aurora_set_point_lights(const float* lights, uint32_t count) {
  aurora::gfx::post_fx::set_point_lights(lights, count);
}
