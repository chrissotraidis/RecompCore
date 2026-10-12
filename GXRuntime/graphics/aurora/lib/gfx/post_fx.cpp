#include "post_fx.hpp"

#include <aurora/aurora.h>

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <array>
#include <atomic>
#include <cstdlib>

namespace aurora::gfx::post_fx {
using webgpu::g_device;

namespace {
std::atomic_bool g_bloom{false};
std::atomic_bool g_farBlur{false};
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

wgpu::RenderPipeline make_pipeline(const char* label, const char* entry, wgpu::TextureFormat format,
                                   uint32_t samples, bool additive) {
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
  const std::array targets{wgpu::ColorTargetState{
      .format = format,
      .blend = additive ? &add : nullptr,
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
    g_bright = make_pipeline("Bloom bright pass", "fs_bright", format, 1, false);
    g_blurH = make_pipeline("Bloom blur (horizontal)", "fs_blur_h", format, 1, false);
    g_blurV = make_pipeline("Bloom blur (vertical)", "fs_blur_v", format, 1, false);
    g_down = make_pipeline("Far blur downsample", "fs_down", format, 1, false);
    g_composite = {};
    g_farComposite = {};
    g_size = {};
  }
  if (!g_composite || samples != g_compositeSamples) {
    g_composite = make_pipeline("Bloom composite", "fs_composite", format, samples, true);
    g_compositeSamples = samples;
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
void draw_far(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView, const wgpu::TextureView& target,
              const wgpu::TextureView& resolve) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = g_farView[0]},
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
      .loadOp = wgpu::LoadOp::Load,
      .storeOp = wgpu::StoreOp::Store,
  }};
  const wgpu::RenderPassDescriptor descriptor{
      .label = "Far blur composite",
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes("Far blur composite"),
  };
  const auto pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(g_farComposite);
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
}

void shutdown() {
  g_bright = g_blurH = g_blurV = g_composite = g_down = g_farComposite = {};
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

bool any_enabled() noexcept { return bloom_enabled() || far_blur_enabled(); }

void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
            const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
            const wgpu::TextureView& resolve, uint32_t msaaSamples) {
  // The far blur needs single-sample depth (the depth buffer is multisampled with MSAA).
  if (far_blur_enabled() && g_module && depthView && msaaSamples == 1 && size.width >= 4 && size.height >= 4) {
    const webgpu::gpu_prof::Zone zone{cmd, "Far blur"};
    ensure_targets(size, format, msaaSamples);
    draw(cmd, "Far blur downsample", g_down, scene, g_farView[0], false);
    for (int i = 0; i < 2; ++i) {
      draw(cmd, "Far blur (horizontal)", g_blurH, g_farView[0], g_farView[1], false);
      draw(cmd, "Far blur (vertical)", g_blurV, g_farView[1], g_farView[0], false);
    }
    draw_far(cmd, depthView, dst, resolve);
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
