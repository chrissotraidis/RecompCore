#include "tex_copy_conv.hpp"

#include "../internal.hpp"
#include "../gx/gx.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"
#include "texture.hpp"
#include "../gx/gx_fmt.hpp"

#include <absl/container/flat_hash_map.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

#include "texture_convert.hpp"

using namespace std::string_literals;

namespace aurora::gfx::tex_copy_conv {
static Module Log("aurora::gfx::tex_copy_conv");

using webgpu::g_device;

static constexpr std::string_view ShaderPreamble = R"(
@group(0) @binding(0) var src_samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;

struct UVTransform {
    offset: vec2f,
    scale: vec2f,
};
@group(0) @binding(2) var<uniform> uv_xf: UVTransform;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);
var<private> uvs: array<vec2f, 3> = array(
    vec2f(0.0, 0.0),
    vec2f(0.0, 2.0),
    vec2f(2.0, 0.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi] * uv_xf.scale + uv_xf.offset;
    return out;
}

fn intensity(rgb: vec3f) -> f32 {
    // ITU-R BT.601 luma coefficients
    return dot(rgb, vec3f(0.257, 0.504, 0.098)) + 16.0 / 255.0;
}

fn quantize4(v: f32) -> f32 {
    return floor(v * 16.0) / 15.0;
}
)"sv;

static const std::string DepthShaderPreamble = R"(
@group(0) @binding(0) var src: texture_depth_2d;

struct UVTransform {
    offset: vec2f,
    scale: vec2f,
};
@group(0) @binding(1) var<uniform> uv_xf: UVTransform;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);
var<private> uvs: array<vec2f, 3> = array(
    vec2f(0.0, 0.0),
    vec2f(0.0, 2.0),
    vec2f(2.0, 0.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi] * uv_xf.scale + uv_xf.offset;
    return out;
}
)"s + (gx::UseReversedZ ? R"(
fn gx_z24(uv: vec2f) -> u32 {
    let texSize = vec2i(textureDimensions(src));
    let coord = clamp(vec2i(floor(uv * vec2f(texSize))), vec2i(0), texSize - vec2i(1));
    let depth = textureLoad(src, coord, 0);
    return min(u32(clamp(1.0 - depth, 0.0, 1.0) * 16777215.0 + 0.5), 0x00ffffffu);
}
)"s
                        : R"(
fn gx_z24(uv: vec2f) -> u32 {
    let texSize = vec2i(textureDimensions(src));
    let coord = clamp(vec2i(floor(uv * vec2f(texSize))), vec2i(0), texSize - vec2i(1));
    let depth = textureLoad(src, coord, 0);
    return min(u32(clamp(depth, 0.0, 1.0) * 16777215.0 + 0.5), 0x00ffffffu);
}
)"s);

// Passthrough blit (for scaling)
static constexpr std::string_view FragPassthrough = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(src, src_samp, in.uv);
}
)"sv;

// GX_TF_I4: 4-bit intensity -> R8Unorm (quantized)
static constexpr std::string_view FragI4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let rgb = textureSample(src, src_samp, in.uv).rgb;
    let i = quantize4(intensity(rgb));
    return vec4f(i, i, i, i);
}
)"sv;

// GX_TF_I8: 8-bit intensity -> R8Unorm
static constexpr std::string_view FragI8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let rgb = textureSample(src, src_samp, in.uv).rgb;
    let i = intensity(rgb);
    return vec4f(i, i, i, i);
}
)"sv;

// GX_TF_I8, keeping the colour beside the intensity: Wind Waker's grey composite copy while Wind Waker HD's monotone
// is drawn (ConvRequest::keepColour). HD doesn't turn that copy grey, so what samples it (the heat haze, the next
// frame's motion blur) keeps the colour; the alpha is the intensity, as the game's.
static constexpr std::string_view FragI8Colour = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let rgb = textureSample(src, src_samp, in.uv).rgb;
    return vec4f(rgb, intensity(rgb));
}
)"sv;

// GX_TF_IA4: 4-bit intensity + 4-bit alpha -> RG8Unorm
static constexpr std::string_view FragIA4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    let i = quantize4(intensity(c.rgb));
    let a = quantize4(c.a);
    return vec4f(i, i, i, a);
}
)"sv;

// GX_TF_IA8: 8-bit intensity + 8-bit alpha -> RG8Unorm
static constexpr std::string_view FragIA8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    let i = intensity(c.rgb);
    return vec4f(i, i, i, c.a);
}
)"sv;

// GX_TF_RGB565: Blit alpha to 1.0
static constexpr std::string_view FragRGB565 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    return vec4f(c.rgb, 1.0);
}
)"sv;

// GX_CTF_R4: 4-bit red -> R8Unorm
static constexpr std::string_view FragR4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = quantize4(textureSample(src, src_samp, in.uv).r);
    return vec4f(r, r, r, r);
}
)"sv;

// GX_CTF_RA4: 4-bit red + 4-bit alpha -> RG8Unorm
static constexpr std::string_view FragRA4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    let r = quantize4(c.r);
    return vec4f(r, r, r, quantize4(c.a));
}
)"sv;

// GX_CTF_RA8: 8-bit red + 8-bit alpha -> RG8Unorm
static constexpr std::string_view FragRA8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    return vec4f(c.r, c.r, c.r, c.a);
}
)"sv;

// GX_CTF_A8: 8-bit alpha -> R8Unorm
static constexpr std::string_view FragA8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let a = textureSample(src, src_samp, in.uv).a;
    return vec4f(a, a, a, a);
}
)"sv;

// GX_CTF_R8: 8-bit red -> R8Unorm
static constexpr std::string_view FragR8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = textureSample(src, src_samp, in.uv).r;
    return vec4f(r, r, r, r);
}
)"sv;

// GX_CTF_G8: 8-bit green -> R8Unorm
static constexpr std::string_view FragG8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let g = textureSample(src, src_samp, in.uv).g;
    return vec4f(g, g, g, g);
}
)"sv;

// GX_CTF_B8: 8-bit blue -> R8Unorm
static constexpr std::string_view FragB8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let b = textureSample(src, src_samp, in.uv).b;
    return vec4f(b, b, b, b);
}
)"sv;

// GX_CTF_RG8: 8-bit red + 8-bit green -> RG8Unorm
static constexpr std::string_view FragRG8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    return vec4f(c.r, c.r, c.r, c.g);
}
)"sv;

// GX_CTF_GB8: 8-bit green + 8-bit blue -> RG8Unorm
static constexpr std::string_view FragGB8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let c = textureSample(src, src_samp, in.uv);
    return vec4f(c.g, c.g, c.g, c.b);
}
)"sv;

// GX_TF_Z16: Upper 16-bits depth -> IA8
static constexpr std::string_view FragZ16 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let z16 = gx_z24(in.uv) >> 8u;
    let i = f32((z16 >> 8u) & 0xFFu) / 255.0;
    let a = f32(z16 & 0xFFu) / 255.0;
    return vec4f(i, i, i, a);
}
)"sv;

// Depth channel selection mirrors Dolphin TextureConverterShaderGen: the 24-bit
// depth expands to bytes (hi, mid, lo) and the copy format picks channels.

// GX_CTF_Z4: Upper 4 bits depth -> R (quantized; the depth preamble has no
// quantize4 helper, so quantize inline: floor(v*16)/15)
static constexpr std::string_view FragZ4 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let v = f32((gx_z24(in.uv) >> 16u) & 0xFFu) / 255.0;
    let r = floor(min(v, 0.999) * 16.0) / 15.0;
    return vec4f(r, r, r, r);
}
)"sv;

// GX_TF_Z8: Upper 8 bits depth -> R
static constexpr std::string_view FragZ8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = f32((gx_z24(in.uv) >> 16u) & 0xFFu) / 255.0;
    return vec4f(r, r, r, r);
}
)"sv;

// GX_CTF_Z8M: Middle 8 bits depth -> R
static constexpr std::string_view FragZ8M = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = f32((gx_z24(in.uv) >> 8u) & 0xFFu) / 255.0;
    return vec4f(r, r, r, r);
}
)"sv;

// GX_CTF_Z8L: Low 8 bits depth -> R
static constexpr std::string_view FragZ8L = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = f32(gx_z24(in.uv) & 0xFFu) / 255.0;
    return vec4f(r, r, r, r);
}
)"sv;

// GX_CTF_Z16L: Low 16 bits depth -> IA8 (mid, low)
static constexpr std::string_view FragZ16L = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let z24 = gx_z24(in.uv);
    let i = f32((z24 >> 8u) & 0xFFu) / 255.0;
    let a = f32(z24 & 0xFFu) / 255.0;
    return vec4f(i, i, i, a);
}
)"sv;

// GX_TF_Z24X8: Full 24-bit depth -> RGB (hi, mid, lo), alpha 1
static constexpr std::string_view FragZ24X8 = R"(
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let z24 = gx_z24(in.uv);
    return vec4f(f32((z24 >> 16u) & 0xFFu) / 255.0,
                 f32((z24 >> 8u) & 0xFFu) / 255.0,
                 f32(z24 & 0xFFu) / 255.0, 1.0);
}
)"sv;

struct ConvPipeline {
  GXTexFmt fmt;
  std::string_view fragShader;
  wgpu::TextureFormat outputFormat;
  const char* label;
};

static constexpr std::array ConvPipelines{
    ConvPipeline{GX_TF_I4, FragI4, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv I4"},
    ConvPipeline{GX_TF_I8, FragI8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv I8"},
    ConvPipeline{GX_TF_IA4, FragIA4, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv IA4"},
    ConvPipeline{GX_TF_IA8, FragIA8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv IA8"},
    ConvPipeline{GX_TF_RGB565, FragRGB565, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv RGB565"},
    ConvPipeline{GX_CTF_R4, FragR4, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv R4"},
    ConvPipeline{GX_CTF_RA4, FragRA4, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv RA4"},
    ConvPipeline{GX_CTF_RA8, FragRA8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv RA8"},
    ConvPipeline{GX_CTF_A8, FragA8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv A8"},
    ConvPipeline{GX_CTF_R8, FragR8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv R8"},
    ConvPipeline{GX_CTF_G8, FragG8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv G8"},
    ConvPipeline{GX_CTF_B8, FragB8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv B8"},
    ConvPipeline{GX_CTF_RG8, FragRG8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv RG8"},
    ConvPipeline{GX_CTF_GB8, FragGB8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv GB8"},
};

static constexpr std::array DepthConvPipelines{
    ConvPipeline{GX_TF_Z16, FragZ16, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv Z16"},
    ConvPipeline{GX_CTF_Z4, FragZ4, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv Z4"},
    ConvPipeline{GX_TF_Z8, FragZ8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv Z8"},
    ConvPipeline{GX_CTF_Z8M, FragZ8M, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv Z8M"},
    ConvPipeline{GX_CTF_Z8L, FragZ8L, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv Z8L"},
    ConvPipeline{GX_CTF_Z16L, FragZ16L, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv Z16L"},
    ConvPipeline{GX_TF_Z24X8, FragZ24X8, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv Z24X8"},
};

static wgpu::BindGroupLayout g_bindGroupLayout;
static wgpu::BindGroupLayout g_depthBindGroupLayout;
static wgpu::Sampler g_nearestSampler;
static wgpu::Sampler g_linearSampler;
static absl::flat_hash_map<GXTexFmt, wgpu::RenderPipeline> g_pipelines;
static wgpu::RenderPipeline g_blitPipeline;
static wgpu::RenderPipeline g_i8ColourPipeline;

static wgpu::RenderPipeline create_pipeline(const ConvPipeline& conv, const std::string_view shaderPreamble,
                                            const wgpu::BindGroupLayout& bindGroupLayout) {
  std::string shaderSource;
  shaderSource.reserve(shaderPreamble.size() + conv.fragShader.size());
  shaderSource += shaderPreamble;
  shaderSource += conv.fragShader;

  const wgpu::ShaderSourceWGSL wgslSource{wgpu::ShaderSourceWGSL::Init{
      .code = shaderSource.c_str(),
  }};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &wgslSource,
      .label = conv.label,
  };
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);

  const std::array colorTargets{wgpu::ColorTargetState{
      .format = conv.outputFormat,
  }};
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };

  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &bindGroupLayout,
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);

  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .label = conv.label,
      .layout = pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = wgpu::PrimitiveTopology::TriangleList,
          },
      .fragment = &fragmentState,
  };
  return g_device.CreateRenderPipeline(&pipelineDescriptor);
}

bool needs_conversion(const GXTexFmt fmt) { return g_pipelines.contains(fmt); }

void initialize() {
  static constexpr std::array bindGroupLayoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler =
              wgpu::SamplerBindingLayout{
                  .type = wgpu::SamplerBindingType::Filtering,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::Float,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Vertex,
          .buffer =
              wgpu::BufferBindingLayout{
                  .type = wgpu::BufferBindingType::Uniform,
              },
      },
  };
  static constexpr wgpu::BindGroupLayoutDescriptor bindGroupLayoutDescriptor{
      .label = "TexCopyConv Bind Group Layout",
      .entryCount = bindGroupLayoutEntries.size(),
      .entries = bindGroupLayoutEntries.data(),
  };
  g_bindGroupLayout = g_device.CreateBindGroupLayout(&bindGroupLayoutDescriptor);

  static constexpr std::array depthBindGroupLayoutEntries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture =
              wgpu::TextureBindingLayout{
                  .sampleType = wgpu::TextureSampleType::Depth,
                  .viewDimension = wgpu::TextureViewDimension::e2D,
              },
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Vertex,
          .buffer =
              wgpu::BufferBindingLayout{
                  .type = wgpu::BufferBindingType::Uniform,
              },
      },
  };
  static constexpr wgpu::BindGroupLayoutDescriptor depthBindGroupLayoutDescriptor{
      .label = "TexCopyConv Depth Bind Group Layout",
      .entryCount = depthBindGroupLayoutEntries.size(),
      .entries = depthBindGroupLayoutEntries.data(),
  };
  g_depthBindGroupLayout = g_device.CreateBindGroupLayout(&depthBindGroupLayoutDescriptor);

  g_blitPipeline = create_pipeline(
      {GX_TF_RGBA8, FragPassthrough, webgpu::g_graphicsConfig.surfaceConfiguration.format, "TexCopyConv Blit"},
      ShaderPreamble, g_bindGroupLayout);
  for (const auto& conv : ConvPipelines) {
    g_pipelines[conv.fmt] = create_pipeline(conv, ShaderPreamble, g_bindGroupLayout);
    if (conv.outputFormat != to_wgpu(conv.fmt)) {
      Log.fatal("Output format mismatch for {}", conv.fmt);
    }
  }
  // Skip depth copies in compatibility mode
  if (webgpu::g_hasCoreFeatures) {
    for (const auto& conv : DepthConvPipelines) {
      g_pipelines[conv.fmt] = create_pipeline(conv, DepthShaderPreamble, g_depthBindGroupLayout);
      if (conv.outputFormat != to_wgpu(conv.fmt)) {
        Log.fatal("Output format mismatch for {}", conv.fmt);
      }
    }
  }

  static constexpr wgpu::SamplerDescriptor nearestSamplerDescriptor{
      .label = "TexCopyConv Nearest Sampler",
      .magFilter = wgpu::FilterMode::Nearest,
      .minFilter = wgpu::FilterMode::Nearest,
  };
  g_nearestSampler = g_device.CreateSampler(&nearestSamplerDescriptor);

  static constexpr wgpu::SamplerDescriptor linearSamplerDescriptor{
      .label = "TexCopyConv Linear Sampler",
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
  };
  g_linearSampler = g_device.CreateSampler(&linearSamplerDescriptor);
}

static void blur_shutdown();

void shutdown() {
  blur_shutdown();
  g_pipelines.clear();
  g_blitPipeline = {};
  g_i8ColourPipeline = {};
  g_bindGroupLayout = {};
  g_depthBindGroupLayout = {};
  g_nearestSampler = {};
  g_linearSampler = {};
}

static void execute(const wgpu::CommandEncoder& cmd, const ConvRequest& req, const wgpu::RenderPipeline& pipeline) {
  if (!pipeline) {
    return;
  }
  wgpu::BindGroup bindGroup;
  if (gx::is_depth_format(req.fmt)) {
    // Skip depth copies in compatibility mode
    if (!webgpu::g_hasCoreFeatures) {
      return;
    }
    const std::array bindGroupEntries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .textureView = req.srcView,
        },
        wgpu::BindGroupEntry{
            .binding = 1,
            .buffer = g_uniformBuffer,
            .offset = req.uniformRange.offset,
            .size = req.uniformRange.size,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .layout = g_depthBindGroupLayout,
        .entryCount = bindGroupEntries.size(),
        .entries = bindGroupEntries.data(),
    };
    bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  } else {
    const auto& sampler = req.sampleFilter == SampleFilter::Linear ? g_linearSampler : g_nearestSampler;
    const std::array bindGroupEntries{
        wgpu::BindGroupEntry{
            .binding = 0,
            .sampler = sampler,
        },
        wgpu::BindGroupEntry{
            .binding = 1,
            .textureView = req.srcView,
        },
        wgpu::BindGroupEntry{
            .binding = 2,
            .buffer = g_uniformBuffer,
            .offset = req.uniformRange.offset,
            .size = req.uniformRange.size,
        },
    };
    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .layout = g_bindGroupLayout,
        .entryCount = bindGroupEntries.size(),
        .entries = bindGroupEntries.data(),
    };
    bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  }

  const std::array colorAttachments{
      wgpu::RenderPassColorAttachment{
          .view = req.dst->attachmentTextureView,
          .loadOp = wgpu::LoadOp::Clear,
          .storeOp = wgpu::StoreOp::Store,
          .clearValue = {0.0, 0.0, 0.0, 0.0},
      },
  };
  const wgpu::RenderPassDescriptor renderPassDescriptor{
      .label = "TexCopyConv Pass",
      .colorAttachmentCount = colorAttachments.size(),
      .colorAttachments = colorAttachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes("EFB copy convert"),
  };
  const auto pass = cmd.BeginRenderPass(&renderPassDescriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bindGroup);
  pass.Draw(3);
  pass.End();
}

void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req) {
  if (req.keepColour && req.fmt == GX_TF_I8) {
    if (!g_i8ColourPipeline)
      g_i8ColourPipeline = create_pipeline(
          ConvPipeline{GX_TF_I8, FragI8Colour, wgpu::TextureFormat::RGBA8Unorm, "TexCopyConv I8 (colour)"},
          ShaderPreamble, g_bindGroupLayout);
    execute(cmd, req, g_i8ColourPipeline);
    return;
  }
  const auto it = g_pipelines.find(req.fmt);
  if (it == g_pipelines.end()) {
    Log.fatal("No copy conversion pipeline for format {}", static_cast<int>(req.fmt));
  }
  execute(cmd, req, it->second);
}

void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req) { execute(cmd, req, g_blitPipeline); }

// --- Blur of a colour copy (the game's depth of field steps) ------------------
// Two passes over a whole-texture triangle, reading texels by position: across
// into one scratch texture, then down into another, mixed there with the sharp
// copy; the result is copied back over the copy.
static constexpr std::string_view BlurShader = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var sharp: texture_2d<f32>;
struct BlurParams {
    sigma: f32,
    mix_amount: f32,
    dir_x: i32,
    dir_y: i32,
};
@group(0) @binding(2) var<uniform> params: BlurParams;

var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    return vec4f(positions[vi], 0.0, 1.0);
}

@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let size = vec2i(textureDimensions(src));
    let p = vec2i(pos.xy);
    let dir = vec2i(params.dir_x, params.dir_y);
    let radius = min(i32(ceil(params.sigma * 3.0)), 48);
    let k = -0.5 / (params.sigma * params.sigma);
    var sum = vec4f(0.0);
    var weights = 0.0;
    for (var i = -radius; i <= radius; i++) {
        let w = exp(f32(i * i) * k);
        sum += textureLoad(src, clamp(p + dir * i, vec2i(0), size - vec2i(1)), 0) * w;
        weights += w;
    }
    var c = sum / weights;
    if (params.dir_y == 1) {
        c = mix(textureLoad(sharp, p, 0), c, params.mix_amount);
    }
    return c;
}
)"sv;

namespace {
struct BlurScratch {
  wgpu::Texture across;
  wgpu::TextureView acrossView;
  wgpu::Texture down;
  wgpu::TextureView downView;
  wgpu::Extent3D size;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
};
wgpu::BindGroupLayout g_blurLayout;
absl::flat_hash_map<uint32_t, wgpu::RenderPipeline> g_blurPipelines; // by wgpu::TextureFormat
BlurScratch g_blurScratch;

const wgpu::RenderPipeline& blur_pipeline(wgpu::TextureFormat format) {
  auto& pipeline = g_blurPipelines[static_cast<uint32_t>(format)];
  if (pipeline) {
    return pipeline;
  }
  if (!g_blurLayout) {
    const std::array entries{
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Fragment,
            .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                                                  .viewDimension = wgpu::TextureViewDimension::e2D},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Fragment,
            .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                                                  .viewDimension = wgpu::TextureViewDimension::e2D},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 2,
            .visibility = wgpu::ShaderStage::Fragment,
            .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Uniform},
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "Copy Blur Bind Group Layout",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_blurLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  }
  const wgpu::ShaderSourceWGSL wgslSource{wgpu::ShaderSourceWGSL::Init{.code = BlurShader.data()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgslSource, .label = "Copy Blur"};
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);
  const std::array colorTargets{wgpu::ColorTargetState{.format = format}};
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = "fs_main",
      .targetCount = colorTargets.size(),
      .targets = colorTargets.data(),
  };
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_blurLayout,
  };
  const wgpu::RenderPipelineDescriptor pipelineDescriptor{
      .label = "Copy Blur",
      .layout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor),
      .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs_main"},
      .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
      .fragment = &fragmentState,
  };
  pipeline = g_device.CreateRenderPipeline(&pipelineDescriptor);
  return pipeline;
}

void blur_scratch(const wgpu::Extent3D& size, wgpu::TextureFormat format) {
  if (g_blurScratch.format == format && g_blurScratch.size.width == size.width &&
      g_blurScratch.size.height == size.height) {
    return;
  }
  const wgpu::TextureDescriptor descriptor{
      .label = "Copy Blur Scratch",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {size.width, size.height, 1},
      .format = format,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  g_blurScratch.across = g_device.CreateTexture(&descriptor);
  g_blurScratch.acrossView = g_blurScratch.across.CreateView();
  g_blurScratch.down = g_device.CreateTexture(&descriptor);
  g_blurScratch.downView = g_blurScratch.down.CreateView();
  g_blurScratch.size = {size.width, size.height, 1};
  g_blurScratch.format = format;
}

void blur_pass(const wgpu::CommandEncoder& cmd, const wgpu::RenderPipeline& pipeline, const wgpu::TextureView& src,
               const wgpu::TextureView& sharp, const wgpu::TextureView& target, float sigma, float mix, bool down) {
  struct Params {
    float sigma;
    float mix;
    int32_t dirX;
    int32_t dirY;
  } const params{sigma, mix, down ? 0 : 1, down ? 1 : 0};
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Copy Blur Params",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(Params),
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(), &params, sizeof(Params));
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .textureView = src},
      wgpu::BindGroupEntry{.binding = 1, .textureView = sharp},
      wgpu::BindGroupEntry{.binding = 2, .buffer = buffer, .size = sizeof(Params)},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_blurLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  const std::array colorAttachments{wgpu::RenderPassColorAttachment{
      .view = target,
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  }};
  const wgpu::RenderPassDescriptor passDescriptor{
      .label = "Copy Blur Pass",
      .colorAttachmentCount = colorAttachments.size(),
      .colorAttachments = colorAttachments.data(),
  };
  const auto pass = cmd.BeginRenderPass(&passDescriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bindGroup);
  pass.Draw(3);
  pass.End();
}
} // namespace

// --- The fast blur (set_fast_blur; DOL_AURORA_GAME_DOF_FAST) ------------------
// The same blur for a small sigma at a fraction of the cost: the copy is taken
// down to half its size with the linear sampler (a 2x2 box), with four side
// taps for what the box and the stretch back leave of the sigma, and that half
// is stretched back over the copy itself with the linear sampler, blended in by
// `mix` (the blend constant). One full-size pass with one sample, against two
// with 2 * ceil(3 sigma) + 1 texel loads each and a copy back.
static constexpr std::string_view FastBlurShader = R"(
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
struct FastBlurParams {
    side: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(2) var<uniform> params: FastBlurParams;

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};
var<private> positions: array<vec2f, 3> = array(
    vec2f(-1.0, 1.0),
    vec2f(-1.0, -3.0),
    vec2f(3.0, 1.0),
);
var<private> uvs: array<vec2f, 3> = array(
    vec2f(0.0, 0.0),
    vec2f(0.0, 2.0),
    vec2f(2.0, 0.0),
);

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4f(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

// Into the half-size target: each sample is the box of 2x2 source texels.
@fragment fn fs_down(in: VertexOutput) -> @location(0) vec4f {
    let two = 2.0 / vec2f(textureDimensions(src));
    var c = textureSampleLevel(src, samp, in.uv, 0.0);
    if (params.side > 0.0) {
        c += params.side * (textureSampleLevel(src, samp, in.uv + vec2f(two.x, 0.0), 0.0) +
                            textureSampleLevel(src, samp, in.uv - vec2f(two.x, 0.0), 0.0) +
                            textureSampleLevel(src, samp, in.uv + vec2f(0.0, two.y), 0.0) +
                            textureSampleLevel(src, samp, in.uv - vec2f(0.0, two.y), 0.0));
        c /= 1.0 + 4.0 * params.side;
    }
    return c;
}

// Over the copy, blended by the constant.
@fragment fn fs_up(in: VertexOutput) -> @location(0) vec4f {
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
)"sv;

namespace {
struct FastBlurScratch {
  wgpu::Texture half;
  wgpu::TextureView halfView;
  wgpu::Extent3D size;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
};
struct FastBlurPipelines {
  wgpu::RenderPipeline down;
  wgpu::RenderPipeline up;
};
wgpu::BindGroupLayout g_fastBlurLayout;
wgpu::Sampler g_fastBlurSampler;
absl::flat_hash_map<uint32_t, FastBlurPipelines> g_fastBlurPipelines; // by wgpu::TextureFormat
FastBlurScratch g_fastBlurScratch;
// -1 until first asked: DOL_AURORA_GAME_DOF_FAST is read then, after the host
// has put its settings in the environment, unless set_fast_blur came first.
std::atomic<int> g_fastBlur{-1};
bool fast_blur_on() {
  int on = g_fastBlur.load(std::memory_order_relaxed);
  if (on < 0) {
    const char* env = std::getenv("DOL_AURORA_GAME_DOF_FAST");
    on = env != nullptr && env[0] != '\0' && env[0] != '0' ? 1 : 0;
    g_fastBlur.store(on, std::memory_order_relaxed);
  }
  return on != 0;
}

const FastBlurPipelines& fast_blur_pipelines(wgpu::TextureFormat format) {
  auto& pipelines = g_fastBlurPipelines[static_cast<uint32_t>(format)];
  if (pipelines.down) {
    return pipelines;
  }
  if (!g_fastBlurLayout) {
    const std::array entries{
        wgpu::BindGroupLayoutEntry{
            .binding = 0,
            .visibility = wgpu::ShaderStage::Fragment,
            .sampler = wgpu::SamplerBindingLayout{.type = wgpu::SamplerBindingType::Filtering},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 1,
            .visibility = wgpu::ShaderStage::Fragment,
            .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::Float,
                                                  .viewDimension = wgpu::TextureViewDimension::e2D},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 2,
            .visibility = wgpu::ShaderStage::Fragment,
            .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Uniform},
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "Copy Fast Blur Bind Group Layout",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_fastBlurLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
    const wgpu::SamplerDescriptor samplerDescriptor{
        .label = "Copy Fast Blur Sampler",
        .addressModeU = wgpu::AddressMode::ClampToEdge,
        .addressModeV = wgpu::AddressMode::ClampToEdge,
        .magFilter = wgpu::FilterMode::Linear,
        .minFilter = wgpu::FilterMode::Linear,
    };
    g_fastBlurSampler = g_device.CreateSampler(&samplerDescriptor);
  }
  const wgpu::ShaderSourceWGSL wgslSource{wgpu::ShaderSourceWGSL::Init{.code = FastBlurShader.data()}};
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgslSource, .label = "Copy Fast Blur"};
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .bindGroupLayoutCount = 1,
      .bindGroupLayouts = &g_fastBlurLayout,
  };
  const auto layout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  {
    const std::array colorTargets{wgpu::ColorTargetState{.format = format}};
    const wgpu::FragmentState fragmentState{
        .module = module,
        .entryPoint = "fs_down",
        .targetCount = colorTargets.size(),
        .targets = colorTargets.data(),
    };
    const wgpu::RenderPipelineDescriptor pipelineDescriptor{
        .label = "Copy Fast Blur Down",
        .layout = layout,
        .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs_main"},
        .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
        .fragment = &fragmentState,
    };
    pipelines.down = g_device.CreateRenderPipeline(&pipelineDescriptor);
  }
  {
    // result = half * constant + copy * (1 - constant), every channel.
    const wgpu::BlendComponent component{
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = wgpu::BlendFactor::Constant,
        .dstFactor = wgpu::BlendFactor::OneMinusConstant,
    };
    const wgpu::BlendState blend{.color = component, .alpha = component};
    const std::array colorTargets{wgpu::ColorTargetState{.format = format, .blend = &blend}};
    const wgpu::FragmentState fragmentState{
        .module = module,
        .entryPoint = "fs_up",
        .targetCount = colorTargets.size(),
        .targets = colorTargets.data(),
    };
    const wgpu::RenderPipelineDescriptor pipelineDescriptor{
        .label = "Copy Fast Blur Up",
        .layout = layout,
        .vertex = wgpu::VertexState{.module = module, .entryPoint = "vs_main"},
        .primitive = wgpu::PrimitiveState{.topology = wgpu::PrimitiveTopology::TriangleList},
        .fragment = &fragmentState,
    };
    pipelines.up = g_device.CreateRenderPipeline(&pipelineDescriptor);
  }
  return pipelines;
}

void fast_blur_scratch(uint32_t width, uint32_t height, wgpu::TextureFormat format) {
  if (g_fastBlurScratch.format == format && g_fastBlurScratch.size.width == width &&
      g_fastBlurScratch.size.height == height) {
    return;
  }
  const wgpu::TextureDescriptor descriptor{
      .label = "Copy Fast Blur Half",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {width, height, 1},
      .format = format,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  g_fastBlurScratch.half = g_device.CreateTexture(&descriptor);
  g_fastBlurScratch.halfView = g_fastBlurScratch.half.CreateView();
  g_fastBlurScratch.size = {width, height, 1};
  g_fastBlurScratch.format = format;
}

void fast_blur_pass(const wgpu::CommandEncoder& cmd, const wgpu::RenderPipeline& pipeline, const wgpu::TextureView& src,
                    const wgpu::TextureView& target, float side, float mix, bool over) {
  struct Params {
    float side;
    float pad[3];
  } const params{side, {0.f, 0.f, 0.f}};
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Copy Fast Blur Params",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(Params),
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(), &params, sizeof(Params));
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_fastBlurSampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = src},
      wgpu::BindGroupEntry{.binding = 2, .buffer = buffer, .size = sizeof(Params)},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_fastBlurLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  const std::array colorAttachments{wgpu::RenderPassColorAttachment{
      .view = target,
      .loadOp = over ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  }};
  const wgpu::RenderPassDescriptor passDescriptor{
      .label = over ? "Copy Fast Blur Up Pass" : "Copy Fast Blur Down Pass",
      .colorAttachmentCount = colorAttachments.size(),
      .colorAttachments = colorAttachments.data(),
  };
  const auto pass = cmd.BeginRenderPass(&passDescriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bindGroup);
  if (over) {
    const wgpu::Color constant{mix, mix, mix, mix};
    pass.SetBlendConstant(&constant);
  }
  pass.Draw(3);
  pass.End();
}

// The fast blur stands in for sigmas its box and stretch can make: in the
// copy's texels, the 2x2 box has a variance of 1/4 and the linear stretch back
// from half size 2/3, so sigmas from about 0.96; the side taps add the rest
// (8 side / (1 + 4 side) texels squared), kept small so the cross stays close
// to round. Light's sigma is 1.
constexpr float FastBlurBase = 0.25f + 2.f / 3.f;
bool fast_blur_fits(const TextureHandle& texture, float sigma, float mix) {
  return fast_blur_on() && texture->attachmentTextureView && mix > 0.f && mix <= 1.f &&
         sigma * sigma >= FastBlurBase && sigma <= 1.25f && texture->size.width >= 4 && texture->size.height >= 4;
}
} // namespace

void set_fast_blur(bool on) { g_fastBlur.store(on ? 1 : 0, std::memory_order_relaxed); }
bool fast_blur() { return fast_blur_on(); }

static void blur_shutdown() {
  g_blurPipelines.clear();
  g_blurLayout = {};
  g_blurScratch = {};
  g_fastBlurPipelines.clear();
  g_fastBlurLayout = {};
  g_fastBlurSampler = {};
  g_fastBlurScratch = {};
}

void blur(const wgpu::CommandEncoder& cmd, const TextureHandle& texture, float sigma, float mix) {
  if (!texture || !texture->texture) {
    return;
  }
  if (fast_blur_fits(texture, sigma, mix)) {
    const webgpu::gpu_prof::Zone zone{cmd, "EFB copy blur (fast)"};
    const auto& pipelines = fast_blur_pipelines(texture->format);
    fast_blur_scratch((texture->size.width + 1) / 2, (texture->size.height + 1) / 2, texture->format);
    const float rest = sigma * sigma - FastBlurBase;
    const float side = rest > 0.f ? rest / (8.f - 4.f * rest) : 0.f;
    fast_blur_pass(cmd, pipelines.down, texture->sampleTextureView, g_fastBlurScratch.halfView, side, mix, false);
    fast_blur_pass(cmd, pipelines.up, g_fastBlurScratch.halfView, texture->attachmentTextureView, 0.f, mix, true);
    return;
  }
  const webgpu::gpu_prof::Zone zone{cmd, "EFB copy blur"};
  sigma = std::max(sigma, 0.05f);
  const auto& pipeline = blur_pipeline(texture->format);
  blur_scratch(texture->size, texture->format);
  blur_pass(cmd, pipeline, texture->sampleTextureView, texture->sampleTextureView, g_blurScratch.acrossView, sigma,
            mix, false);
  blur_pass(cmd, pipeline, g_blurScratch.acrossView, texture->sampleTextureView, g_blurScratch.downView, sigma, mix,
            true);
  const wgpu::TexelCopyTextureInfo from{.texture = g_blurScratch.down};
  const wgpu::TexelCopyTextureInfo to{.texture = texture->texture};
  const wgpu::Extent3D extent{texture->size.width, texture->size.height, 1};
  cmd.CopyTextureToTexture(&from, &to, &extent);
}

} // namespace aurora::gfx::tex_copy_conv
