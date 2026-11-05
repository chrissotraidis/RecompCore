#include "gxcore_draw.hpp"

#include <aurora/aurora.h> // aurora_set_forced_anisotropy

#include "../webgpu/gpu.hpp"
#include "../gx/gx.hpp" // UseReversedZ + set_logical_viewport (substrate glue)
#include "frame_interp.hpp"
#include "pipeline_cache.hpp"
#include "post_fx.hpp" // the sun shadow map
#include "soft_effects.hpp"
#include "smooth_normal_cache.hpp"
#include "thread_cpu.hpp"
#include "texture.hpp"
#include "tex_copy_conv.hpp" // EFB-copy format conversion (63/S16)
#include "texture_replacement.hpp" // Dolphin-format HD texture packs

#include <gxruntime/gxcore/gxcore.hpp> // EfbCopyCommand
#include <gxruntime/gxcore/texture_decode.hpp>
#include <gxruntime/gxcore/texture_encode.hpp>

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

namespace aurora::gfx::gxcore {

namespace gxc = gxruntime::gxcore;

using webgpu::g_device;
using webgpu::g_graphicsConfig;

static Module Log("aurora::gfx::gxcore");

namespace {

// Device-retaining caches, hoisted out of their functions so shutdown() can
// release them BEFORE webgpu::shutdown()/window::shutdown() destroy the Vulkan
// instance and XCB connection. As function-local statics they were destroyed at
// exit(), when the last wgpu::Device ref's destructor walked a freed XCB
// connection to destroy fences -> "double free or corruption (!prev)".
std::mutex g_texBindLayoutMutex;
absl::flat_hash_map<uint32_t, wgpu::BindGroupLayout> g_texBindLayoutCache;
wgpu::Device g_texBindLayoutOwner;

std::mutex g_emptyTexmapMutex;
wgpu::Device g_emptyTexmapOwner;
wgpu::Texture g_emptyTexmapTexture;
wgpu::TextureView g_emptyTexmapView;

wgpu::CompareFunction to_compare(gxc::CompareMode func) {
  // GC compare flipped for the substrate's reversed-Z (gx/gx.cpp:526 shape).
  switch (func) {
  case gxc::CompareMode::Never:
    return wgpu::CompareFunction::Never;
  case gxc::CompareMode::Less:
    return gx::UseReversedZ ? wgpu::CompareFunction::Greater
                            : wgpu::CompareFunction::Less;
  case gxc::CompareMode::Equal:
    return wgpu::CompareFunction::Equal;
  case gxc::CompareMode::LEqual:
    return gx::UseReversedZ ? wgpu::CompareFunction::GreaterEqual
                            : wgpu::CompareFunction::LessEqual;
  case gxc::CompareMode::Greater:
    return gx::UseReversedZ ? wgpu::CompareFunction::Less
                            : wgpu::CompareFunction::Greater;
  case gxc::CompareMode::NEqual:
    return wgpu::CompareFunction::NotEqual;
  case gxc::CompareMode::GEqual:
    return gx::UseReversedZ ? wgpu::CompareFunction::LessEqual
                            : wgpu::CompareFunction::GreaterEqual;
  case gxc::CompareMode::Always:
  default:
    return wgpu::CompareFunction::Always;
  }
}

wgpu::BlendFactor to_blend_factor_src(gxc::SrcBlendFactor factor,
                                      bool dual_source) {
  switch (factor) {
  case gxc::SrcBlendFactor::Zero:
    return wgpu::BlendFactor::Zero;
  case gxc::SrcBlendFactor::One:
    return wgpu::BlendFactor::One;
  case gxc::SrcBlendFactor::DstClr:
    return wgpu::BlendFactor::Dst;
  case gxc::SrcBlendFactor::InvDstClr:
    return wgpu::BlendFactor::OneMinusDst;
  case gxc::SrcBlendFactor::SrcAlpha:
    return dual_source ? wgpu::BlendFactor::Src1Alpha
                       : wgpu::BlendFactor::SrcAlpha;
  case gxc::SrcBlendFactor::InvSrcAlpha:
    return dual_source ? wgpu::BlendFactor::OneMinusSrc1Alpha
                       : wgpu::BlendFactor::OneMinusSrcAlpha;
  case gxc::SrcBlendFactor::DstAlpha:
    return wgpu::BlendFactor::DstAlpha;
  case gxc::SrcBlendFactor::InvDstAlpha:
  default:
    return wgpu::BlendFactor::OneMinusDstAlpha;
  }
}

wgpu::BlendFactor to_blend_factor_dst(gxc::DstBlendFactor factor,
                                      bool dual_source) {
  switch (factor) {
  case gxc::DstBlendFactor::Zero:
    return wgpu::BlendFactor::Zero;
  case gxc::DstBlendFactor::One:
    return wgpu::BlendFactor::One;
  case gxc::DstBlendFactor::SrcClr:
    return wgpu::BlendFactor::Src;
  case gxc::DstBlendFactor::InvSrcClr:
    return wgpu::BlendFactor::OneMinusSrc;
  case gxc::DstBlendFactor::SrcAlpha:
    return dual_source ? wgpu::BlendFactor::Src1Alpha
                       : wgpu::BlendFactor::SrcAlpha;
  case gxc::DstBlendFactor::InvSrcAlpha:
    return dual_source ? wgpu::BlendFactor::OneMinusSrc1Alpha
                       : wgpu::BlendFactor::OneMinusSrcAlpha;
  case gxc::DstBlendFactor::DstAlpha:
    return wgpu::BlendFactor::DstAlpha;
  case gxc::DstBlendFactor::InvDstAlpha:
  default:
    return wgpu::BlendFactor::OneMinusDstAlpha;
  }
}

// Forced anisotropy (aurora_set_forced_anisotropy): 1 leaves the game's samplers.
std::uint16_t initial_forced_anisotropy() {
  const char* env = std::getenv("DOL_AURORA_FORCE_ANISO");
  const long samples = env != nullptr ? std::strtol(env, nullptr, 10) : 1L;
  return static_cast<std::uint16_t>(std::clamp(samples, 1L, 16L));
}
std::atomic<std::uint16_t> g_forcedAnisotropy{initial_forced_anisotropy()};

wgpu::AddressMode to_address_mode(std::uint8_t wrap) {
  switch (wrap) {
  case 1:
    return wgpu::AddressMode::Repeat;
  case 2:
    return wgpu::AddressMode::MirrorRepeat;
  case 0:
  default:
    return wgpu::AddressMode::ClampToEdge;
  }
}

// The levels a draw's texture has of its own: an HD replacement's mip chain
// (decoded GameCube textures are uploaded with one level).
uint32_t replacement_mips(const TextureHandle& texture) noexcept {
  return texture && texture->isReplacement ? texture->mipCount : 1u;
}

// `textureMips`: replacement_mips() of the texture the sampler goes with.
wgpu::SamplerDescriptor sampler_descriptor(const gxc::PlanSampler& sampler, uint32_t textureMips = 1u) {
  // Most of Wind Waker's textures have no mips (TX_SETMODE0 filter 4), so their
  // sampler stays on level 0. An HD replacement is several times their size and
  // brings its own mips: drawn from level 0 alone it aliases at a distance, and
  // shimmers and flickers as the camera turns. Like Dolphin (a custom texture's
  // max LOD is 255), a replacement's levels are all used, blended linearly when
  // the game asked for no mip filter.
  const bool ownMips = textureMips > 1u;
  const bool mipmaps = sampler.mipmap_filter != 0u || ownMips;
  std::uint16_t maxAnisotropy = 1;
  if (mipmaps && (sampler.max_aniso == 1u || sampler.max_aniso == 2u)) {
    maxAnisotropy = sampler.max_aniso == 1u
                        ? std::max<std::uint16_t>(
                              webgpu::g_graphicsConfig.textureAnisotropy / 2u,
                              1u)
                        : std::max<std::uint16_t>(
                              webgpu::g_graphicsConfig.textureAnisotropy, 1u);
  }
  // The player's forced anisotropy (aurora_set_forced_anisotropy), with
  // Dolphin's rule: every texture except one filtered nearest both ways (pixel
  // art, fonts), all filters linear. Most of Wind Waker's textures have no
  // mips (TX_SETMODE0 filter 4), so the anisotropic taps are what sharpen the
  // ground and the walls at a glancing angle.
  const std::uint16_t forced = g_forcedAnisotropy.load(std::memory_order_relaxed);
  if (forced > 1u && (sampler.min_filter != 0u || sampler.mag_filter != 0u))
    maxAnisotropy = std::max(maxAnisotropy, forced);
  auto magFilter = sampler.mag_filter != 0u ? wgpu::FilterMode::Linear
                                             : wgpu::FilterMode::Nearest;
  auto minFilter = sampler.min_filter != 0u ? wgpu::FilterMode::Linear
                                             : wgpu::FilterMode::Nearest;
  auto mipFilter = sampler.mipmap_filter == 2u || (ownMips && sampler.mipmap_filter == 0u)
                       ? wgpu::MipmapFilterMode::Linear
                       : wgpu::MipmapFilterMode::Nearest;
  if (maxAnisotropy > 1u) {
    magFilter = wgpu::FilterMode::Linear;
    minFilter = wgpu::FilterMode::Linear;
    mipFilter = wgpu::MipmapFilterMode::Linear;
  }
  return {
      .label = "GXCore Sampler",
      .addressModeU = to_address_mode(sampler.wrap_s),
      .addressModeV = to_address_mode(sampler.wrap_t),
      .addressModeW = wgpu::AddressMode::Repeat,
      .magFilter = magFilter,
      .minFilter = minFilter,
      .mipmapFilter = mipFilter,
      .lodMinClamp = sampler.mipmap_filter != 0u ? static_cast<float>(sampler.min_lod) / 16.f : 0.f,
      .lodMaxClamp = ownMips    ? static_cast<float>(textureMips - 1u)
                     : mipmaps ? static_cast<float>(sampler.max_lod) / 16.f
                               : 0.f,
      .maxAnisotropy = maxAnisotropy,
  };
}

// Texture bind group layout for a used-texmap set (63/Mfin multi-texmap): texmap
// t occupies binding 2t (texture) + 2t+1 (sampler), matching the WGSL. Cached per
// mask; used_mask=1 (texmap 0 only) reproduces the pre-Mfin single-texmap layout.
wgpu::BindGroupLayout texture_bind_group_layout(uint32_t used_mask = 1u) {
  // Both the pipeline compiler and FIFO submission use this cache. A lookup
  // concurrent with flat_hash_map growth can read an invalid layout handle.
  std::lock_guard lock{g_texBindLayoutMutex};
  // Retaining the owner also prevents pointer reuse from matching a dead
  // device after renderer reinitialization.
  if (g_texBindLayoutOwner.Get() != g_device.Get()) {
    g_texBindLayoutCache.clear();
    g_texBindLayoutOwner = g_device;
  }
  auto it = g_texBindLayoutCache.find(used_mask);
  if (it != g_texBindLayoutCache.end())
    return it->second;
  std::vector<wgpu::BindGroupLayoutEntry> entries;
  for (uint32_t t = 0; t < 8u; ++t) {
    if ((used_mask & (1u << t)) == 0u)
      continue;
    entries.push_back(wgpu::BindGroupLayoutEntry{
        .binding = 2u * t,
        .visibility = wgpu::ShaderStage::Fragment,
        .texture =
            wgpu::TextureBindingLayout{
                .sampleType = wgpu::TextureSampleType::Float,
                .viewDimension = wgpu::TextureViewDimension::e2D,
            },
    });
    entries.push_back(wgpu::BindGroupLayoutEntry{
        .binding = 2u * t + 1u,
        .visibility = wgpu::ShaderStage::Fragment,
        .sampler =
            wgpu::SamplerBindingLayout{
                .type = wgpu::SamplerBindingType::Filtering,
            },
    });
  }
  const wgpu::BindGroupLayoutDescriptor descriptor{
      .label = "GXCore Texture Bind Group Layout",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  auto layout = g_device.CreateBindGroupLayout(&descriptor);
  g_texBindLayoutCache.emplace(used_mask, layout);
  return layout;
}

// Wind Waker HD's character light (gxc ShaderKey::hd_actor_light): a flagged pipeline's group 0 is the pass's
// static group (its two storage buffers, bindings 0 and 1, which the shader doesn't read) and HD's light: its
// uniform (gxc::HdActorLightConstants), the shadow pre-pass and its sampler (bindings 2 to 4), HD's cloud shadow
// texture and its sampler (5 and 6; post_fx::actor_cloud_source(), else a white texel), and HD's light buffer's
// lights (7, gxc::HdPointLights; none in the off group). Made on first use by
// the pipeline compiler or the render thread, under the texture layouts' lock.
wgpu::BindGroupLayout g_actorLightLayout;
wgpu::Device g_actorLightLayoutOwner;

wgpu::BindGroupLayout actor_light_group_layout() {
  std::lock_guard lock{g_texBindLayoutMutex};
  if (g_actorLightLayout && g_actorLightLayoutOwner.Get() == g_device.Get())
    return g_actorLightLayout;
  const std::array entries{
      wgpu::BindGroupLayoutEntry{
          .binding = 0,
          .visibility = wgpu::ShaderStage::Vertex,
          .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::ReadOnlyStorage},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Vertex,
          .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::ReadOnlyStorage},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Uniform,
                                              .minBindingSize = sizeof(gxc::HdActorLightConstants)},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 3,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::Float,
                                                .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 4,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = wgpu::SamplerBindingLayout{.type = wgpu::SamplerBindingType::Filtering},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 5,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = wgpu::TextureBindingLayout{.sampleType = wgpu::TextureSampleType::Float,
                                                .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 6,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = wgpu::SamplerBindingLayout{.type = wgpu::SamplerBindingType::Filtering},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 7,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = wgpu::BufferBindingLayout{.type = wgpu::BufferBindingType::Uniform,
                                              .minBindingSize = sizeof(gxc::HdPointLights)},
      },
  };
  const wgpu::BindGroupLayoutDescriptor descriptor{
      .label = "GXCore actor light (HD) group",
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  g_actorLightLayout = g_device.CreateBindGroupLayout(&descriptor);
  g_actorLightLayoutOwner = g_device;
  return g_actorLightLayout;
}

// Guest-identity texture cache (S13 A3). Keyed by the texture's guest identity
// AND the TLUT identity it indexes: the same CI image bytes re-palettized to a
// different TLUT decode to different pixels, so the palette address/format/
// entries are part of the key. 63/Mfin adds `content_hash` (xxh3 of the actual
// source texels + TLUT): the live game reuses one guest texture-buffer address
// for different images across screens (stadium select overwrites the captain
// art) and streams movie frames through a single buffer, so identity alone would
// bind the stale decode. Hashing the content makes a content change under an
// unchanged identity a distinct entry (Dolphin/Aurora texel-hash model). Static
// replay fixtures never rewrite an address, so the no-reconvert property holds.
struct TextureKey {
  uint32_t address;
  uint32_t size;
  uint32_t format;
  uint32_t width;
  uint32_t height;
  uint32_t tlut_address;
  uint32_t tlut_format;
  uint32_t tlut_entries;
  uint64_t content_hash;
  bool operator==(const TextureKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const TextureKey& key) {
    return H::combine(std::move(h), key.address, key.size, key.format,
                      key.width, key.height, key.tlut_address, key.tlut_format,
                      key.tlut_entries, key.content_hash);
  }
};

absl::flat_hash_map<TextureKey, TextureHandle> g_textureCache;
// Original textures drawn while their HD replacement decodes on a background
// thread. They stay out of g_textureCache so the next draw asks again; the
// entry goes when the replacement arrives (or with the cache's clears).
absl::flat_hash_map<TextureKey, TextureHandle> g_textureAwaitingReplacement;
// address -> the key currently cached at that guest address. When new content
// arrives at an address (movie streaming, screen transitions), the prior entry
// for that address is evicted so the content-hashed cache stays bounded to one
// live entry per buffer instead of accumulating every historical frame.
struct TextureAddressState {
  TextureKey key;
  uint64_t texel_epoch = 0;
  uint64_t tlut_epoch = 0;
  bool texel_epoch_valid = false;
  bool tlut_epoch_valid = false;
  // Small textures only (see kAlwaysRehashBytes): hash of the texels and
  // palette as last decoded, and the frame in which they were last compared.
  uint64_t small_hash = 0;
  uint64_t verified_frame = ~0ull;
  // The texture cached under `key`, so an unchanged texture takes one lookup
  // instead of two (the second, by the whole key, was a tenth of the FIFO
  // worker's time on Outset); empty once that entry is evicted.
  TextureHandle handle;
};
absl::flat_hash_map<uint32_t, TextureAddressState> g_textureAddrKey;
// Counts presented frames; a small texture is re-hashed at most once in each.
std::atomic<uint64_t> g_textureVerifyFrame{0};
// A small texture is re-hashed even when its dirty epoch has not moved: the
// game rewrites some in place with plain CPU stores that no dirty mark sees
// (Wind Waker's A/B action labels, 80x24 IA4 at 0x01673420 and 0x01674C60,
// showed "Charts" for "Choose"/"Return" in the pause menu). Dolphin reads
// memory coherently, so the reference shows the new text. Once a frame per
// texture, up to 4 KB of texels plus the palette.
constexpr uint32_t kAlwaysRehashBytes = 4096u;
uint64_t small_texture_hash(const uint8_t* bytes, uint32_t size,
                            const void* tlut, uint32_t tlut_size) {
  uint64_t h = XXH3_64bits(bytes, size);
  if (tlut != nullptr && tlut_size != 0u)
    h ^= XXH3_64bits(tlut, tlut_size) * 0x9E3779B97F4A7C15ull;
  return h;
}
TextureCacheStats g_textureCacheStats;
TextureDirtyEpochObserver g_textureDirtyEpochObserver = nullptr;
// For EFB copies (guest_memory_dirty.h's write epoch): unset, the texture one.
TextureDirtyEpochObserver g_efbCopyEpochObserver = nullptr;

bool efb_copy_epoch(uint32_t address, uint32_t size, uint64_t* epoch) {
  const auto observer = g_efbCopyEpochObserver != nullptr ? g_efbCopyEpochObserver : g_textureDirtyEpochObserver;
  return observer != nullptr && observer(address, size, epoch);
}

// EFB copy allocations mirror GXState::copyTextureCache: one guest destination
// may be reused with different dimensions or formats, which require distinct GPU
// textures. g_efbCopyTextures mirrors GXState::copyTextures and selects the most
// recent allocation when that guest destination is sampled.
struct EfbCopyKey {
  uint32_t address;
  uint32_t width;
  uint32_t height;
  uint32_t format;
  bool opaque;  // the view reads alpha as one (EFB without alpha)
  bool operator==(const EfbCopyKey&) const = default;
  template <typename H>
  friend H AbslHashValue(H h, const EfbCopyKey& key) {
    return H::combine(std::move(h), key.address, key.width, key.height,
                      key.format, key.opaque);
  }
};
absl::flat_hash_map<EfbCopyKey, TextureHandle> g_efbCopyCache;
struct EfbCopyBinding {
  TextureHandle handle;
  uint32_t byte_size = 0;
  uint64_t memory_epoch = 0;
  bool memory_epoch_valid = false;
};
absl::flat_hash_map<uint32_t, EfbCopyBinding> g_efbCopyTextures;
// The grey composite's copy's second texture while monotone, per copy (copy_efb_to_texture()), and the bindings it
// stands in for until the frame's end (note_frame_presented()).
absl::flat_hash_map<EfbCopyKey, TextureHandle> g_monotoneHazeTextures;
struct MonotoneHazeRestore {
  uint32_t address = 0;
  EfbCopyBinding copy;
  TextureHandle haze;
};
std::vector<MonotoneHazeRestore> g_monotoneHazeRestore;

} // namespace

namespace {
// Whether the alpha test is statically guaranteed to keep every pixel (Mirror
// AlphaTest::TestResult enough to tell).
bool alpha_test_always_passes(const gxc::ShaderKey& shader) {
  const bool c0_never = shader.alpha_comp0 == 0u;
  const bool c1_never = shader.alpha_comp1 == 0u;
  const bool c0_always = shader.alpha_comp0 == 7u;
  const bool c1_always = shader.alpha_comp1 == 7u;
  switch (shader.alpha_logic) {
  case 0u: // And
    return c0_always && c1_always;
  case 1u: // Or
    return c0_always || c1_always;
  case 2u: // Xor
    return (c0_always && c1_never) || (c0_never && c1_always);
  case 3u: // Xnor
    return (c0_always && c1_always) || (c0_never && c1_never);
  }
  return false;
}

// A draw the sun shadow map draws again (DrawData::sunCaster): opaque scenery
// and characters. Leaves, fences and grass (alpha-tested: the map has no
// textures, so they would cast solid cards), what is blended (water, clouds,
// particles, the game's own blob shadows), what writes no depth (the sky) and
// lines and points are left out.
bool sun_caster(const gxc::PipelineKey& key) {
  // DOL_AURORA_SUN_SHADOW_MAP_ALL=1: alpha-tested draws cast too, as solid
  // shapes (local test: which shadows the alpha-test exclusion is costing).
  static const bool s_all = [] {
    const char* value = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_ALL");
    return value != nullptr && value[0] == '1';
  }();
  return key.depth_test != 0u && key.depth_update != 0u && key.color_update != 0u && key.blend_enable == 0u &&
         key.blend_subtract == 0u && key.primitive_topology == 0u &&
         (s_all || alpha_test_always_passes(key.shader));
}

// Whether the alpha test is statically guaranteed to drop every pixel (a
// depth-priming draw): such a draw shows nothing, so it casts nothing.
bool alpha_test_always_fails(const gxc::ShaderKey& shader) {
  const bool c0_never = shader.alpha_comp0 == 0u;
  const bool c1_never = shader.alpha_comp1 == 0u;
  const bool c0_always = shader.alpha_comp0 == 7u;
  const bool c1_always = shader.alpha_comp1 == 7u;
  switch (shader.alpha_logic) {
  case 0u: // And
    return c0_never || c1_never;
  case 1u: // Or
    return c0_never && c1_never;
  case 2u: // Xor
    return (c0_always && c1_always) || (c0_never && c1_never);
  case 3u: // Xnor
    return (c0_always && c1_never) || (c0_never && c1_always);
  }
  return false;
}

// A draw the sun shadow map draws again with its texture's cutout
// (DrawData::sunCasterAlpha): solid scenery whose alpha test drops pixels
// (roofs, awnings, a tree's leaves, fences). Its caster pipeline samples the
// texture and discards below half, so it casts its true silhouette.
bool sun_caster_alpha(const gxc::PipelineKey& key) {
  return key.depth_test != 0u && key.depth_update != 0u && key.color_update != 0u && key.blend_enable == 0u &&
         key.blend_subtract == 0u && key.primitive_topology == 0u && key.shader.textured != 0u &&
         !alpha_test_always_passes(key.shader) && !alpha_test_always_fails(key.shader);
}

// A blended draw that casts with its texture's cutout all the same (part of
// DrawData::sunCasterAlpha, with submit_draw_plan()'s checks that its positions
// stand still and it is no character): alpha blended (SRCALPHA, INVSRCALPHA),
// writing depth and alpha-tested, as the GameCube draws its grass and bushes so
// they fade near the camera (Outset's tufts, texture 0037EE00). Wind Waker HD
// draws the same tufts opaque and alpha-tested and casts them, cut out, in all
// of its cascades (bridge_frame14789: 764 instances x 3 cascades). Additive
// glows (ONE), what writes no depth, and what moves (the sea's crests and the
// wake, which the game writes each frame) are left out.
// DOL_AURORA_SUN_SHADOW_MAP_BLENDED_CUTOUT=0 leaves them all out, to compare.
bool sun_caster_blended_cutout(const gxc::PipelineKey& key) {
  static const bool s_on = [] {
    const char* value = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_BLENDED_CUTOUT");
    return value == nullptr || value[0] != '0';
  }();
  return s_on && key.depth_test != 0u && key.depth_update != 0u && key.color_update != 0u && key.blend_enable != 0u &&
         key.blend_subtract == 0u && key.src_factor == static_cast<uint8_t>(gxc::SrcBlendFactor::SrcAlpha) &&
         key.dst_factor == static_cast<uint8_t>(gxc::DstBlendFactor::InvSrcAlpha) && key.primitive_topology == 0u &&
         key.shader.textured != 0u && !alpha_test_always_passes(key.shader) && !alpha_test_always_fails(key.shader);
}

// The sea's surface, which the solid caster rule takes (it is opaque and
// writes depth): strips the game lays around the player in world space, about
// 51,000 units across, and flat quads out to 450,000 for the distance, flat
// but for the waves. Nothing else is that wide and that flat. As in Wind Waker
// HD it takes shadows but casts none: in the map a swell's crest shaded the
// trough behind it, and a tower standing there lost the foot of its shadow.
// DOL_AURORA_SUN_SHADOW_MAP_SEA_CASTS=1 puts it back in the map.
// Measured at the size it is drawn: the sea rooms' square of sea past the
// grid's edge is a 100-unit quad its position matrix scales 1,000 times, which
// by its own vertices (100 across) wasn't the sea, so the HD look's brightness
// took the open sea to a pale cyan.
bool sea_shape(const gxc::DrawPlan& plan) {
  if (plan.constants.projection[3][3] == 1.f || plan.vertices.size() < 3)
    return false;
  float lo[3] = {plan.vertices[0], plan.vertices[1], plan.vertices[2]};
  float hi[3] = {lo[0], lo[1], lo[2]};
  for (size_t v = gxc::kVertexFloats; v + 2 < plan.vertices.size(); v += gxc::kVertexFloats)
    for (int c = 0; c < 3; ++c) {
      lo[c] = std::min(lo[c], plan.vertices[v + c]);
      hi[c] = std::max(hi[c], plan.vertices[v + c]);
    }
  // Each model axis's scale: the length of the position matrix's column for it.
  const auto& m = plan.constants.posnormalmatrix;
  float scale[3];
  for (int c = 0; c < 3; ++c)
    scale[c] = std::sqrt(m[0][c] * m[0][c] + m[1][c] * m[1][c] + m[2][c] * m[2][c]);
  const float wide = std::max((hi[0] - lo[0]) * scale[0], (hi[2] - lo[2]) * scale[2]);
  return wide >= 20000.f && (hi[1] - lo[1]) * scale[1] <= 0.02f * wide;
}

bool sea_surface(const gxc::DrawPlan& plan) {
  static const bool s_seaCasts = [] {
    const char* value = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_SEA_CASTS");
    return value != nullptr && value[0] == '1';
  }();
  return !s_seaCasts && sea_shape(plan);
}

// The sea's textures, by the texture packs' hash of their bytes (Dolphin's
// tex1_WxH_hash_fmt): Always.arc's b_sea_tex0and2.bti, the open sea's water
// and white foam lines (d_a_sea.cpp binds it to texmaps 0 and 2; models'
// B_Sea share it); the island rooms' ring of sea, Txa_umi_kiwa_01 and
// Txa_umi_kage_01; and their beaches' surf, Txa_nami_01, Txa_nami_02,
// Txa_nami_ia and Txa_namisoko_ia (the wet band under it); and still and
// running water, the Private Oasis's pool (Txa_dummy_umi) and the rivers' and
// falls' edge, flow and spray (Txa_test_kiwa_a, _nagare_j, _sibuki_a: the
// oasis, Hyrule, the Forest Haven), which came out pale cyan brightened. Kept
// per guest address, checked again when the bytes there change.
bool sea_texture(uint32_t address, uint32_t format, uint32_t width, uint32_t height, const void* data,
                 uint32_t available) {
  struct SeaTexture {
    uint32_t format, width, height;
    uint64_t hash;
  };
  static constexpr SeaTexture kSea[] = {
      {0, 256, 256, 0x35b39510e63e0eddull}, // b_sea_tex0and2, B_Sea (I4)
      {0, 128, 64, 0xae4b9cf3ade885aeull},  // Txa_umi_kiwa_01
      {0, 128, 64, 0xa75016cd8d47c1f9ull},  // Txa_umi_kage_01
      {0, 128, 128, 0xcb795d7577fa2a1eull}, // Txa_nami_01
      {0, 128, 128, 0x6c7d2c9e6e7e34adull}, // Txa_nami_02
      {2, 128, 128, 0x8d4cd80958bc6cd6ull}, // Txa_nami_ia (IA4)
      {2, 128, 128, 0xb485ef6b2fed6f20ull}, // Txa_namisoko_ia
      {14, 128, 128, 0xb8e02ad23e1b98b9ull}, // Txa_dummy_umi (CMPR)
      {0, 128, 128, 0xb26b325c4b3c1a25ull}, // Txa_test_kiwa_a
      {0, 128, 128, 0x6d34c29c2f22bc6dull}, // Txa_test_nagare_j
      {0, 128, 128, 0xa1248d29670925e0ull}, // Txa_test_sibuki_a
  };
  bool shape = false;
  for (const SeaTexture& sea : kSea)
    shape = shape || (sea.format == format && sea.width == width && sea.height == height);
  // I4 and CMPR are 4 bits a texel, IA4 8 (these sizes fill whole tiles).
  const uint32_t bytes = format == 2u ? width * height : width * height / 2;
  if (!shape || data == nullptr || available < bytes)
    return false;
  struct Seen {
    uint64_t head = 0;
    bool sea = false;
  };
  static std::unordered_map<uint32_t, Seen> s_seen;
  const uint64_t head = XXH3_64bits(data, 64);
  auto [it, fresh] = s_seen.try_emplace(address);
  if (fresh || it->second.head != head) {
    const uint64_t hash = XXH64(data, bytes, 0);
    bool sea = false;
    for (const SeaTexture& texture : kSea)
      sea = sea || (texture.format == format && texture.width == width && texture.height == height &&
                    texture.hash == hash);
    it->second = Seen{head, sea};
  }
  return it->second.sea;
}

// A sea draw: one that samples the sea's textures, or the sea rooms' own sea, an
// untextured solid square of the environment's sea colour under d_a_sea's
// grid (100,000 units across, at the sea's height), seen past the grid's edge.
bool sea_draw(const gxc::DrawPlan& plan) {
  if (plan.constants.projection[3][3] == 1.f)
    return false;
  bool sea = false;
  if (plan.texmap_mask > 1u && (plan.texmap_mask & (plan.texmap_mask - 1u)) != 0u) {
    for (uint32_t t = 0; t < 8 && !sea; ++t) {
      const auto& texture = plan.textures[t];
      sea = (plan.texmap_mask >> t & 1u) != 0u && texture.valid &&
            sea_texture(texture.address, texture.format, texture.width, texture.height, texture.data,
                        texture.available);
    }
  } else if (plan.has_texture) {
    sea = sea_texture(plan.tex_address, plan.tex_format, plan.tex_width, plan.tex_height, plan.tex_data,
                      plan.tex_available);
  } else {
    sea = plan.pipeline.depth_update != 0u && plan.pipeline.blend_enable == 0u && sea_shape(plan);
  }
  return sea;
}

// A sea draw for the sea mask (post_fx.hpp), while it is wanted (sea_mask_wanted()).
bool sea_mask_draw(const gxc::DrawPlan& plan) {
  return post_fx::sea_mask_wanted() && sea_draw(plan);
}

// A glow model (Dragon Roost Cavern's lava geysers): blended, and its colour from none of the stages' texture
// colours, only the registers and constants, or the rasterized colour where its textures cut its shape (their alpha
// alone is used: a geyser's pale core, half the game's light added to its colour): what is painted, it
// isn't. Untextured, a colour from the rasterized one is lit scenery's.
static bool glow_model(const gxc::PipelineKey& key) {
  const auto& sh = key.shader;
  if (sh.tev_valid == 0u || sh.num_tev_stages == 0u)
    return false;
  bool ras = false, cut = false;
  for (uint32_t i = 0; i < sh.num_tev_stages && i < gxc::kMaxTevStages; ++i) {
    const auto& t = sh.tev_stages[i];
    for (const uint8_t arg : {t.cc_a, t.cc_b, t.cc_c, t.cc_d}) {
      if (arg == 8u) // TEXC
        return false;
      ras = ras || arg == 10u || arg == 11u; // RASC, RASA
    }
    for (const uint8_t arg : {t.ac_a, t.ac_b, t.ac_c, t.ac_d})
      cut = cut || (arg == 4u && t.tevorders_enable != 0u); // TEXA
  }
  return !ras || cut;
}

// An effect, which the HD look's sun shadows, occlusion and cloud shadows leave
// as it is (post_fx.hpp effect_mask_target()): a draw blended over what is
// behind it that writes no depth (the spin attack's rings, the Master Sword's
// glow, sparks, smoke), so the depth buffer under it is the scenery's behind,
// whose shade it is not in. Wind Waker HD draws these after its lighting.
// One that writes depth is an effect too when it is one of the game's particles (the host's tag: the dust a
// shutting door raises, drawn alpha-tested into the depth and then in colour over it) or a glow model (glow_model());
// its depth hides the scenery's under it, so the HD look shades there from the sea mask's depth, which it leaves out
// (effect_depth_draw(), post_fx.hpp shading_depth()).
static bool effect_draw_depth(const gxc::DrawPlan& plan) {
  return plan.pipeline.depth_update == 0u || plan.draw_tag != 0u || glow_model(plan.pipeline);
}

bool effect_draw(const gxc::DrawPlan& plan) {
  const auto& key = plan.pipeline;
  // Left out: the GameCube's own shadows (drawn through the frame buffer's
  // alpha, the destination's alpha their blend factor), which darken the ground
  // they are on; and what has no depth test (the sky's clouds), whose pixels
  // are those the draws after it leave, which the mask, drawn against the
  // finished depth, can't tell (the sky's clouds lit up shadows on the land
  // drawn over them).
  if (key.blend_enable == 0u || key.blend_subtract != 0u || !effect_draw_depth(plan) || key.depth_test == 0u ||
      key.color_update == 0u || key.src_factor == 6u || key.src_factor == 7u ||
      plan.constants.projection[3][3] == 1.f)
    return false;
  // The sea's own blended draws (the surf and the ring of sea around an
  // island) are the sea's surface, shaded as it is.
  if (sea_draw(plan))
    return false;
  // A character's own decals (its eyes and mouth, blended over its face by its
  // skinned model) are on the face and take its shade; what a skinned model
  // adds as light (the Master Sword's glow) doesn't.
  if (key.shader.has_pos_mtx_idx != 0u && key.dst_factor != 1u)
    return false;
  return true;
}

// An effect that writes depth (effect_draw()).
bool effect_depth_draw(const gxc::DrawPlan& plan) { return plan.pipeline.depth_update != 0u && effect_draw(plan); }
} // namespace

// Toon-shaded: a texgen from a lit colour (GX_TG_SRTG: Wind Waker's characters' ramp, Link's hair included).
static bool toon_draw(const gxc::PipelineKey& key) {
  for (uint32_t i = 0; i < key.shader.num_tex_gens && i < gxc::kMaxTexGens; ++i) {
    const auto& tg = key.shader.tex_gens[i];
    if (tg.enabled != 0u && (tg.texgentype == static_cast<uint8_t>(gxc::TexGenType::Color0) ||
                             tg.texgentype == static_cast<uint8_t>(gxc::TexGenType::Color1)))
      return true;
  }
  return false;
}

// Toon-shaded (toon_draw(): Link's hair included, which is rigid on his head) or skinned: a character for the sea
// mask.
static bool character_draw(const gxc::PipelineKey& key) { return key.shader.has_pos_mtx_idx != 0u || toon_draw(key); }

// DOL_AURORA_SUN_SHADOW_SKINNED_SCENERY=1: the sun pass takes a skinned draw that isn't toon-shaded (a palm, whose
// leaves sway) as scenery, as Wind Waker HD draws its palms with the terrain's shader.
static bool skinned_scenery() {
  static const bool value = [] {
    const char* env = std::getenv("DOL_AURORA_SUN_SHADOW_SKINNED_SCENERY");
    return env != nullptr && env[0] == '1';
  }();
  return value;
}

// DOL_AURORA_TERRAIN_NORMAL_LOG=1 (local debug): whether the scenery the sun shadows' facing term shades
// carries vertex normals and lights with them, a line per distinct vertex format, lighting and texture.
static void log_terrain_normals(const gxc::PipelineKey& key, uint32_t texture, uint32_t texmapMask) {
  static const bool on = std::getenv("DOL_AURORA_TERRAIN_NORMAL_LOG") != nullptr;
  if (!on)
    return;
  static std::mutex mutex;
  static std::set<std::array<uint32_t, 4>> seen;
  const auto& c0 = key.shader.litchan[0];
  const auto& a0 = key.shader.litchan[2];
  const uint32_t format = key.shader.has_vertex_normal | key.shader.has_vertex_binormal << 1 |
                          key.shader.has_pos_mtx_idx << 2 | key.shader.lit_valid << 3 |
                          key.shader.num_color_chans << 4 | key.shader.has_color0 << 8;
  const uint32_t light = c0.enablelighting | c0.matsource << 1 | c0.ambsource << 2 | c0.diffusefunc << 3 |
                         c0.attnfunc << 5 | static_cast<uint32_t>(c0.light_mask) << 8 |
                         static_cast<uint32_t>(a0.enablelighting) << 16 | static_cast<uint32_t>(a0.light_mask) << 24;
  std::lock_guard lock(mutex);
  if (!seen.insert({format, light, texture, texmapMask}).second)
    return;
  std::fprintf(stderr,
               "[terrain-normal] tex=%08x texmaps=%02x normal=%u nbt=%u skinned=%u lit_valid=%u chans=%u color0=%u | "
               "c0 light=%u mat=%u amb=%u diff=%u attn=%u mask=%02x | a0 light=%u mask=%02x\n",
               texture, texmapMask, key.shader.has_vertex_normal, key.shader.has_vertex_binormal,
               key.shader.has_pos_mtx_idx, key.shader.lit_valid, key.shader.num_color_chans, key.shader.has_color0,
               c0.enablelighting, c0.matsource, c0.ambsource, c0.diffusefunc, c0.attnfunc, c0.light_mask,
               a0.enablelighting, a0.light_mask);
}

// DOL_AURORA_ACTOR_LIGHT_LOG=1 (local test): each toon-shaded shader once, as its pipeline is made: its texgens,
// its first colour channel and every TEV stage, to see which TEV patterns Wind Waker's characters use (Wind Waker
// HD's actor light replaces the ramp's mix of C0 and K0; the pattern decides what is the ramp, what is the albedo).
static void log_toon_pipeline(const gxc::PipelineKey& key) {
  static const bool on = std::getenv("DOL_AURORA_ACTOR_LIGHT_LOG") != nullptr;
  if (!on || !toon_draw(key))
    return;
  static std::mutex mutex;
  static std::set<uint64_t> seen;
  uint64_t hash = 1469598103934665603ull;
  const auto* bytes = reinterpret_cast<const uint8_t*>(&key.shader);
  for (size_t i = 0; i < sizeof key.shader; ++i)
    hash = (hash ^ bytes[i]) * 1099511628211ull;
  std::lock_guard lock(mutex);
  if (!seen.insert(hash).second)
    return;
  const auto& s = key.shader;
  const auto& c0 = s.litchan[0];
  std::string line = "[actor-light] toon " + std::to_string(seen.size()) +
                     " skinned=" + std::to_string(s.has_pos_mtx_idx) + " normal=" + std::to_string(s.has_vertex_normal) +
                     " dstalpha=" + std::to_string(s.use_dst_alpha) + " c0(light=" + std::to_string(c0.enablelighting) +
                     " mat=" + std::to_string(c0.matsource) + " amb=" + std::to_string(c0.ambsource) +
                     " diff=" + std::to_string(c0.diffusefunc) + " attn=" + std::to_string(c0.attnfunc) +
                     " mask=" + std::to_string(c0.light_mask) + ") texgens";
  for (uint32_t i = 0; i < s.num_tex_gens && i < gxc::kMaxTexGens; ++i)
    line += " " + std::to_string(i) + ":" + std::to_string(s.tex_gens[i].texgentype) + "/" +
            std::to_string(s.tex_gens[i].sourcerow);
  line += " | " + std::to_string(s.num_tev_stages) + " stages";
  for (uint32_t i = 0; i < s.num_tev_stages && i < gxc::kMaxTevStages; ++i) {
    const auto& t = s.tev_stages[i];
    char stage[160];
    std::snprintf(stage, sizeof stage, " [%u tc%u map%u ch%u en%u kc%u c=%u,%u,%u,%u b%u o%u s%u ->%u a=%u,%u,%u,%u]", i,
                  t.tevorders_texcoord, t.tevorders_texmap, t.tevorders_colorchan, t.tevorders_enable, t.ksel_kc,
                  t.cc_a, t.cc_b, t.cc_c, t.cc_d, t.cc_bias, t.cc_op, t.cc_scale, t.cc_dest, t.ac_a, t.ac_b, t.ac_c,
                  t.ac_d);
    line += stage;
  }
  std::fprintf(stderr, "%s\n", line.c_str());
}

bool needs_early_depth_emulation(const gxc::PipelineKey& key) {
  if (key.depth_test == 0u || key.depth_update == 0u ||
      key.early_depth_test == 0u || key.shader.tev_valid == 0u) {
    return false;
  }
  // Fail and undetermined both need an early depth write; the ordinary color
  // shader will discard failures afterward.
  return !alpha_test_always_passes(key.shader);
}

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config) {
  const gxc::PipelineKey& key = config.key;
  // depthOnly 2: the ubershader for this fixed-function state (the key's
  // shader is empty but for destination alpha), drawing any shader key.
  const bool uber = config.depthOnly == 2u;
  const bool depthOnly = config.depthOnly == 1u;
  CHECK(key.shader.use_dst_alpha == 0 ||
            webgpu::g_dualSourceBlendingSupported,
        "GX destination alpha requires WebGPU dual-source blending");
  if (!uber && !depthOnly)
    log_toon_pipeline(key);
  std::string wgsl = uber ? gxc::generate_uber_wgsl(key.shader.use_dst_alpha != 0) : gxc::generate_wgsl(key.shader);
  if (depthOnly) {
    wgsl += "\n@fragment\nfn fs_depth_only() -> @location(0) vec4f {\n"
            "    return vec4f(0.0);\n}\n";
  }
  wgpu::ShaderSourceWGSL sourceDescriptor{};
  sourceDescriptor.code = wgsl.c_str();
  const wgpu::ShaderModuleDescriptor moduleDescriptor{
      .nextInChain = &sourceDescriptor,
      .label = "GXCore Shader Module",
  };
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);

  // Group 0 keeps the pass preamble's static bind group compatible (the shader
  // never references it); 1 = shared dynamic VS uniform. On the TEV path (S14)
  // 2 = shared dynamic PS uniform and 3 = texture; else 2 = texture. Putting
  // the PS uniform before the texture keeps an untextured TEV draw gap-free.
  const bool tev = key.shader.tev_valid != 0;
  const bool textured = key.shader.textured != 0;
  // Multi-texmap: the texture group's layout matches the set of texmaps the WGSL
  // declares (derived identically from the shader key), so pipeline and bind
  // group agree. Untextured draws never use the texture group (layoutCount below).
  const uint32_t tex_mask = uber ? 0xFFu : textured ? gxc::used_texmap_mask(key.shader) : 1u;
  // Wind Waker HD's character light: group 0 longer, with its uniform and shadow (actor_light_group_layout()).
  // HD's light buffer on the lit scenery (hd_point_light) takes the same group 0 for its lights.
  const bool actorLight = !uber && !depthOnly && (key.shader.hd_actor_light != 0u || key.shader.hd_point_light != 0u);
  std::array<wgpu::BindGroupLayout, 4> bindGroupLayouts{
      actorLight ? actor_light_group_layout() : g_staticBindGroupLayout,
      g_uniformBindGroupLayout,
      tev || uber ? g_uniformBindGroupLayout : texture_bind_group_layout(tex_mask),
      texture_bind_group_layout(tex_mask),
  };
  const size_t layoutCount =
      uber ? 4 : depthOnly ? 2 : tev ? (textured ? 4 : 3) : (textured ? 3 : 2);
  const wgpu::PipelineLayoutDescriptor layoutDescriptor{
      .label = "GXCore Pipeline Layout",
      .bindGroupLayoutCount = layoutCount,
      .bindGroupLayouts = bindGroupLayouts.data(),
  };
  const auto pipelineLayout = g_device.CreatePipelineLayout(&layoutDescriptor);

  // Fixed decoded-vertex layout (gxruntime/gxcore/shader.hpp). The normal
  // (location 8) is only declared by the lit shader, so add it to the pipeline
  // only when the key is lit — WGSL requires the vertex layout to satisfy every
  // shader input.
  std::vector<wgpu::VertexAttribute> attributes{
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x3,
          .offset = gxc::kVertexPosOffset,
          .shaderLocation = 0,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Uint32,
          .offset = gxc::kVertexPosMtxOffset,
          .shaderLocation = 1,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x4,
          .offset = gxc::kVertexColor0Offset,
          .shaderLocation = 2,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x4,
          .offset = gxc::kVertexColor1Offset,
          .shaderLocation = 3,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset,
          .shaderLocation = 4,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 8,
          .shaderLocation = 5,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 16,
          .shaderLocation = 6,
      },
      wgpu::VertexAttribute{
          .format = wgpu::VertexFormat::Float32x2,
          .offset = gxc::kVertexUvOffset + 24,
          .shaderLocation = 7,
      },
  };
  // Item-5 texgen inputs must mirror the generator's VertexIn: emboss needs the
  // NBT normal/binormal/tangent + light dir; a Color1/emboss key that is unlit
  // still declares the normal (location 8). Detect emboss demand from the key.
  bool has_emboss = false;
  bool has_normal_source = false;
  for (std::uint32_t i = 0; i < key.shader.num_tex_gens; ++i) {
    const auto type =
        static_cast<gxc::TexGenType>(key.shader.tex_gens[i].texgentype);
    if (type == gxc::TexGenType::EmbossMap)
      has_emboss = true;
    if (type == gxc::TexGenType::Regular &&
        static_cast<gxc::TexSourceRow>(key.shader.tex_gens[i].sourcerow) ==
            gxc::TexSourceRow::Normal)
      has_normal_source = true;
  }
  // Mirror the generator exactly: locations 8/10/11 are declared only when the
  // vertex FORMAT carries that attribute. A lit/emboss draw whose format omits
  // it reads the cached fallback from the uniform instead (I_CACHED_NORMAL), so
  // the shader does not declare the input and the layout must not provide it.
  // A vertex with TEX5..7 and no normal carries them in these three slots
  // (gxc::kMaxRawTexCoords), so the shader declares all three.
  const bool tex_hi = key.shader.raw_tex_hi_in_nbt != 0u;
  if (uber || tex_hi ||
      ((key.shader.lit_valid != 0 || has_emboss || has_normal_source) &&
       key.shader.has_vertex_normal != 0) ||
      (key.shader.hd_point_light & 3u) == 2u) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexNormalOffset,
        .shaderLocation = 8,
    });
  }
  if (uber || key.shader.has_tex_mtx_idx != 0) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Uint32,
        .offset = gxc::kVertexTexMtxIdxOffset,
        .shaderLocation = 9,
    });
  }
  if (uber || tex_hi || (has_emboss && key.shader.has_vertex_binormal != 0)) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexBinormalOffset,
        .shaderLocation = 10,
    });
  }
  if (uber || tex_hi || (has_emboss && key.shader.has_vertex_tangent != 0)) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x3,
        .offset = gxc::kVertexTangentOffset,
        .shaderLocation = 11,
    });
  }
  if (uber || (key.shader.uv_mask & (1u << 4u)) != 0u) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Float32x2,
        .offset = gxc::kVertexUvOffset + 32u,
        .shaderLocation = 12,
    });
  }
  if (uber || (key.shader.has_tex_mtx_idx != 0 &&
                (key.shader.tex_mtx_idx_mask & 0xF0u) != 0u)) {
    attributes.push_back(wgpu::VertexAttribute{
        .format = wgpu::VertexFormat::Uint32,
        .offset = gxc::kVertexTexMtxIdxHiOffset,
        .shaderLocation = 13,
    });
  }
  const wgpu::VertexBufferLayout vertexLayout{
      .arrayStride = gxc::kVertexStrideBytes,
      .stepMode = wgpu::VertexStepMode::Vertex,
      .attributeCount = attributes.size(),
      .attributes = attributes.data(),
  };

  // The ubershader draws without early-depth emulation (a frame or two).
  const bool emulateEarlyDepth = !uber && needs_early_depth_emulation(key);
  const bool depthCompare = key.depth_test != 0;
  const bool softTarget = config.softTarget != 0u;
  const wgpu::DepthStencilState depthStencil{
      .format = g_graphicsConfig.depthFormat,
      .depthWriteEnabled =
          depthOnly ? depthCompare && key.depth_update != 0
                    : depthCompare && key.depth_update != 0 &&
                          !emulateEarlyDepth,
      .depthCompare =
          emulateEarlyDepth && !depthOnly
              ? wgpu::CompareFunction::Equal
              : depthCompare ? to_compare(static_cast<gxc::CompareMode>(
                                   key.depth_func))
                             : wgpu::CompareFunction::Always,
  };

  // GC subtract mode forces ONE/ONE with dst - src (Dolphin RenderState).
  const bool dual_source = key.shader.use_dst_alpha != 0;
  wgpu::BlendState blendState{};
  if (key.blend_subtract != 0) {
    blendState.color = {
        .operation = wgpu::BlendOperation::ReverseSubtract,
        .srcFactor = wgpu::BlendFactor::One,
        .dstFactor = wgpu::BlendFactor::One,
    };
    blendState.alpha = dual_source
                           ? wgpu::BlendComponent{
                                 .operation = wgpu::BlendOperation::Add,
                                 .srcFactor = wgpu::BlendFactor::One,
                                 .dstFactor = wgpu::BlendFactor::Zero,
                             }
                           : blendState.color;
  } else {
    blendState.color = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = to_blend_factor_src(
            static_cast<gxc::SrcBlendFactor>(key.src_factor), dual_source),
        .dstFactor = to_blend_factor_dst(
            static_cast<gxc::DstBlendFactor>(key.dst_factor), dual_source),
    };
    blendState.alpha = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = to_blend_factor_src(
            static_cast<gxc::SrcBlendFactor>(key.src_factor_alpha),
            dual_source),
        .dstFactor = to_blend_factor_dst(
            static_cast<gxc::DstBlendFactor>(key.dst_factor_alpha),
            dual_source),
    };
  }
  if (softTarget) {
    // The soft effects' buffer accumulates coverage (soft_effects.hpp).
    blendState.alpha = {
        .operation = wgpu::BlendOperation::Add,
        .srcFactor = wgpu::BlendFactor::One,
        .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha,
    };
  }
  auto writeMask = wgpu::ColorWriteMask::None;
  if (softTarget)
    writeMask = wgpu::ColorWriteMask::All;
  else if (key.color_update != 0) {
    writeMask |= wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green |
                 wgpu::ColorWriteMask::Blue;
  }
  if (key.alpha_update != 0 && !softTarget)
    writeMask |= wgpu::ColorWriteMask::Alpha;
  const bool blending =
      key.blend_enable != 0 || key.blend_subtract != 0;
  const wgpu::ColorTargetState colorTarget{
      .format = g_graphicsConfig.surfaceConfiguration.format,
      .blend = blending ? &blendState : nullptr,
      .writeMask = writeMask,
  };
  const wgpu::ColorTargetState depthOnlyTarget{
      .format = g_graphicsConfig.surfaceConfiguration.format,
      .blend = nullptr,
      .writeMask = wgpu::ColorWriteMask::None,
  };
  const wgpu::FragmentState fragmentState{
      .module = module,
      .entryPoint = depthOnly ? "fs_depth_only" : "fs_main",
      .targetCount = 1,
      .targets = depthOnly ? &depthOnlyTarget : &colorTarget,
  };

  auto cullMode = wgpu::CullMode::None;
  switch (static_cast<gxc::CullMode>(key.cull_mode)) {
  case gxc::CullMode::Back:
    cullMode = wgpu::CullMode::Back;
    break;
  case gxc::CullMode::Front:
    cullMode = wgpu::CullMode::Front;
    break;
  default:
    break; // None here; All was skipped at plan time
  }
  auto topology = wgpu::PrimitiveTopology::TriangleList;
  if (key.primitive_topology == 1u) {
    topology = wgpu::PrimitiveTopology::LineList;
    cullMode = wgpu::CullMode::None;
  } else if (key.primitive_topology == 2u) {
    topology = wgpu::PrimitiveTopology::PointList;
    cullMode = wgpu::CullMode::None;
  }

  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "GXCore Pipeline",
      .layout = pipelineLayout,
      .vertex =
          wgpu::VertexState{
              .module = module,
              .entryPoint = "vs_main",
              .bufferCount = 1,
              .buffers = &vertexLayout,
          },
      .primitive =
          wgpu::PrimitiveState{
              .topology = topology,
              // Substrate winding convention (gx/gx.cpp to_primitive_state).
              .frontFace = wgpu::FrontFace::CW,
              .cullMode = cullMode,
          },
      .depthStencil = softTarget ? nullptr : &depthStencil,
      .multisample =
          wgpu::MultisampleState{
              .count = config.msaaSamples,
          },
      .fragment = &fragmentState,
  };
  if (!uber)
    return g_device.CreateRenderPipeline(&descriptor);
  // An ubershader pipeline takes most of a second when its shader is not in
  // Dawn's blob cache yet (pipeline_cache.cpp makes one first): say so.
  const auto start = std::chrono::steady_clock::now();
  auto made = g_device.CreateRenderPipeline(&descriptor);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  if (ms >= 100.0)
    std::fprintf(stderr, "[pipelines] the ubershader compiled in %.0f ms (%s)\n", ms,
                 key.shader.use_dst_alpha != 0 ? "dual-source blending" : "plain");
  return made;
}

// What render() last bound in the pass being encoded, so a draw sets only
// what differs from the draw before it. Every draw set its vertex and index
// ranges and its bind groups again, six or seven commands a draw, and the
// in-between frames encode every draw once more each (three times at 120 Hz):
// the render worker's encoding was most of a game frame in a busy scene. The
// vertex and index buffers are bound whole once (a draw's range is its base
// vertex and first index), and a bind group is set again only when it or its
// offset changed or the pipeline did (a tev pipeline's group 2 is another
// layout than an untextured one's).
namespace {
// While a run is drawn into the soft effects' buffer (set_soft_target).
static bool g_softTarget = false;

struct PassState {
  WGPUBuffer vertexBuffer = nullptr; // bound whole, or a particle's range
  uint64_t vertexOffset = 0;
  bool indexBound = false;
  PipelineRef pipeline = 0;
  WGPUBindGroup group1 = nullptr;
  uint32_t offset1 = UINT32_MAX;
  WGPUBindGroup group2 = nullptr;
  uint32_t offset2 = UINT32_MAX;
  WGPUBindGroup group3 = nullptr;
};
PassState g_pass;
} // namespace

void reset_pass_state() { g_pass = PassState{}; }

static std::atomic_bool g_efficientEffects{false};
bool efficient_effects() noexcept { return g_efficientEffects.load(std::memory_order_relaxed); }
void set_soft_target(bool on) noexcept { g_softTarget = on; }
static const wgpu::TextureView& empty_texmap_view();

// Wind Waker HD's character light, for the pass being encoded (begin_actor_light_pass()): the group 0 its flagged
// draws bind (actor_light_group_layout()), or null for the off group (params.x 0: the GameCube result), which
// passes that don't set one up take.
namespace {
wgpu::BindGroup g_actorPassGroup;
wgpu::BindGroup g_actorOffGroup;
wgpu::Sampler g_actorSampler;

// DOL_AURORA_ACTOR_LIGHT_DEBUG: 1 the ramp coordinate s, 2 the shadow, 3 the ramp, 4 the hemisphere, 5 the rim,
// 6 the cloud.
float actor_light_debug() {
  static const float value = [] {
    const char* env = std::getenv("DOL_AURORA_ACTOR_LIGHT_DEBUG");
    return env != nullptr ? static_cast<float>(std::clamp(std::atoi(env), 0, 6)) : 0.f;
  }();
  return value;
}

// DOL_AURORA_ACTOR_LIGHT_RIM=intensity,exponent,back: HD's rim (its defaults for every character: 2, 5, 0.75).
const std::array<float, 3>& actor_light_rim() {
  static const std::array<float, 3> value = [] {
    std::array<float, 3> rim{2.f, 5.f, 0.75f};
    if (const char* env = std::getenv("DOL_AURORA_ACTOR_LIGHT_RIM")) {
      float a, b, c;
      if (std::sscanf(env, "%f,%f,%f", &a, &b, &c) == 3)
        rim = {a, b, c};
    }
    return rim;
  }();
  return value;
}

wgpu::BindGroup make_actor_group(const gxc::HdActorLightConstants& constants, const wgpu::TextureView& shadow,
                                 const wgpu::TextureView& cloud = {}, const wgpu::Sampler& cloudSampler = {},
                                 const gxc::HdPointLights& lights = {}) {
  if (!g_actorSampler) {
    const wgpu::SamplerDescriptor sampler{
        .label = "GXCore actor light (HD) shadow",
        .addressModeU = wgpu::AddressMode::ClampToEdge,
        .addressModeV = wgpu::AddressMode::ClampToEdge,
        .magFilter = wgpu::FilterMode::Linear,
        .minFilter = wgpu::FilterMode::Linear,
        .mipmapFilter = wgpu::MipmapFilterMode::Linear,
    };
    g_actorSampler = g_device.CreateSampler(&sampler);
  }
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "GXCore actor light (HD)",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(constants),
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, sizeof(constants)), &constants, sizeof(constants));
  buffer.Unmap();
  // Only the lights in use are copied (the rest of the buffer is zero).
  const size_t lightBytes = sizeof(lights);
  const wgpu::BufferDescriptor lightDescriptor{
      .label = "GXCore point lights (HD)",
      .usage = wgpu::BufferUsage::Uniform,
      .size = lightBytes,
      .mappedAtCreation = true,
  };
  const auto lightBuffer = g_device.CreateBuffer(&lightDescriptor);
  std::memcpy(lightBuffer.GetMappedRange(0, lightBytes), &lights, lightBytes);
  lightBuffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .buffer = g_vertexBuffer},
      wgpu::BindGroupEntry{.binding = 1, .buffer = g_storageBuffer},
      wgpu::BindGroupEntry{.binding = 2, .buffer = buffer, .offset = 0, .size = sizeof(constants)},
      wgpu::BindGroupEntry{.binding = 3, .textureView = shadow},
      wgpu::BindGroupEntry{.binding = 4, .sampler = g_actorSampler},
      wgpu::BindGroupEntry{.binding = 5, .textureView = cloud ? cloud : empty_texmap_view()},
      wgpu::BindGroupEntry{.binding = 6, .sampler = cloudSampler ? cloudSampler : g_actorSampler},
      wgpu::BindGroupEntry{.binding = 7, .buffer = lightBuffer, .offset = 0, .size = lightBytes},
  };
  const wgpu::BindGroupDescriptor descriptor{
      .label = "GXCore actor light (HD) group",
      .layout = actor_light_group_layout(),
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  return g_device.CreateBindGroup(&descriptor);
}

const wgpu::BindGroup& actor_light_pass_group() {
  if (g_actorPassGroup)
    return g_actorPassGroup;
  if (!g_actorOffGroup)
    g_actorOffGroup = make_actor_group(gxc::HdActorLightConstants{}, empty_texmap_view());
  return g_actorOffGroup;
}

void release_actor_light() {
  g_actorPassGroup = {};
  g_actorOffGroup = {};
  g_actorSampler = {};
}
} // namespace

bool begin_actor_light_pass(const post_fx::Camera* camera, const post_fx::SunMap* sunMap,
                            const wgpu::Extent3D& size) {
  g_actorPassGroup = {};
  if (!post_fx::actor_light_hd_switch() || camera == nullptr || !camera->valid || size.width == 0 ||
      size.height == 0)
    return false;
  gxc::HdActorLightConstants c{};
  if (!post_fx::actor_light_palette(c.c0, c.shadow, c.k))
    return false;
  c.own_k[3] = post_fx::actor_light_own(c.own_c0, c.own_k) ? 1.f : 0.f;
  // The shadow: this frame's pre-pass of the characters' own where common.cpp made it for this pass
  // (post_fx::draw_actor_shadow(), HD's mip 1), else last frame's blurred pre-pass where the post pass made one; how
  // strongly the shadow map's light shows; none: lit (1).
  wgpu::TextureView shadow;
  float lod = 0.f;
  bool sameFrame = false;
  const bool haveShadow =
      sunMap != nullptr && sunMap->valid && post_fx::actor_shadow_source(shadow, lod, sameFrame);
  c.params[0] = 1.f;
  c.params[1] = actor_light_debug();
  c.params[2] = lod;
  c.params[3] = haveShadow ? std::clamp(sunMap->fade, 0.f, 1.f) : 0.f;
  c.c0[3] = haveShadow ? 1.f : 0.f;
  // The light the shadow map is drawn from (the sun, the moon at night: HD's rule), else the game's sun; to the
  // camera's view space (the inverse view's rotation, transposed).
  const float* v = camera->view.data();
  const float world[3] = {sunMap != nullptr && sunMap->valid ? sunMap->light[0] : v[17],
                          sunMap != nullptr && sunMap->valid ? sunMap->light[1] : v[18],
                          sunMap != nullptr && sunMap->valid ? sunMap->light[2] : v[19]};
  float light[3];
  for (int j = 0; j < 3; ++j)
    light[j] = v[j] * world[0] + v[4 + j] * world[1] + v[8 + j] * world[2];
  const float length = std::sqrt(light[0] * light[0] + light[1] * light[1] + light[2] * light[2]);
  for (int j = 0; j < 3; ++j)
    c.light[j] = length > 1e-6f ? light[j] / length : 0.f;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      c.rows[i][j] = v[i * 4 + j];
  const auto& rim = actor_light_rim();
  c.rim[0] = rim[0];
  c.rim[1] = rim[1];
  c.rim[2] = rim[2];
  // HD's cloud shadow, from the scenery's (post_fx::actor_cloud_source()); none: the cloud is 1 (rim.w 0).
  wgpu::TextureView cloud;
  wgpu::Sampler cloudSampler;
  const bool haveCloud = post_fx::actor_cloud_source(*camera, c.cloud, cloud, cloudSampler);
  c.rim[3] = haveCloud ? 1.f : 0.f;
  c.screen[0] = 1.f / static_cast<float>(size.width);
  c.screen[1] = 1.f / static_cast<float>(size.height);
  // The GameCube's effect light on characters (the sword-hit flash, bombs, the item get: a TEV stage the game adds
  // only while one reaches the model) on HD's colour, while HD's light buffer is off: with it on, the same effect
  // lights reach the characters through it (below), as in HD, whose toon shader has no such term, and the stage
  // would count them twice. DOL_AURORA_ACTOR_LIGHT_HD_EFFECT=1 keeps the stage, 0 leaves it out, either way.
  static const int effectSwitch = [] {
    const char* env = std::getenv("DOL_AURORA_ACTOR_LIGHT_HD_EFFECT");
    return env == nullptr || env[0] == '\0' ? -1 : env[0] != '0' ? 1 : 0;
  }();
  const bool pointLights = post_fx::point_lights_hd_active();
  c.screen[2] = (effectSwitch >= 0 ? effectSwitch == 1 : !pointLights) ? 1.f : 0.f;
  // Wind Waker HD's light buffer (post_fx::point_lights_hd_active()): the frame's lights, to the camera's view space
  // (world - eye through the inverse view's rotation, transposed).
  gxc::HdPointLights lights{};
  if (pointLights) {
    const uint32_t count = std::min<uint32_t>(camera->lightCount, gxc::kMaxHdPointLights);
    const float eye[3] = {v[3], v[7], v[11]};
    for (uint32_t i = 0; i < count; ++i) {
      const float* l = camera->lights.data() + i * 8;
      const float d[3] = {l[0] - eye[0], l[1] - eye[1], l[2] - eye[2]};
      for (int j = 0; j < 3; ++j)
        lights.pos_r[i][j] = v[j] * d[0] + v[4 + j] * d[1] + v[8 + j] * d[2];
      lights.pos_r[i][3] = l[3];
      for (int j = 0; j < 3; ++j)
        lights.colour[i][j] = std::max(l[4 + j], 0.f);
    }
    lights.count[0] = static_cast<float>(count);
    lights.count[1] = gxc::kHdPointLightActor;
    lights.count[2] = gxc::kHdPointLightScenery;
  }
  g_actorPassGroup =
      make_actor_group(c, haveShadow ? shadow : empty_texmap_view(), cloud, cloudSampler, lights);
  return true;
}

void end_actor_light_pass() { g_actorPassGroup = {}; }


// What a draw, or one of a batch's (submit_draw_plan) draws, is encoded with:
// its constants and a particle's in-between vertices (empty: the frame's
// own), its indices, and its first vertex from the batch's.
namespace {
struct DrawPart {
  Range uniform;
  Range verts;
  Range pixel;
  uint32_t firstIndex;
  uint32_t indexCount;
  uint32_t firstVertex;
};
} // namespace

namespace {
// The parts render() encodes a draw as, with the constants and vertices of the
// frame being encoded (an in-between frame's step, or the frame's own).
void collect_parts(const DrawData& data, std::vector<DrawPart>& parts) {
  // The in-between frame being encoded: its step's block and vertices (a
  // traced frame's own, matched while recording, are one step's).
  const bool interpolated = frame_interp::encoding_interpolated();
  const int step = interpolated ? interp_replay_step() : 0;
  const auto job_ranges = [&](uint32_t job, Range& uniform, Range& verts, Range& pixel) {
    uniform = verts = pixel = {};
    if (!interpolated)
      return;
    if (data.interpJob != UINT32_MAX) {
      const InterpRanges& ranges = interp_job_ranges(job);
      uniform = ranges.uniform[step];
      verts = ranges.verts[step];
      pixel = ranges.pixel[step];
    } else if (step == 0) {
      uniform = data.interpUniformRange;
      verts = data.interpVertRange;
    }
  };
  const uint32_t firstIndex = data.idxRange.offset / sizeof(uint16_t);
  // One draw, or a batch as one; a batch whose draws' in-between blocks (or a
  // particle's vertices, laid out apart from the batch's) differ, one by one.
  parts.clear();
  DrawPart whole{.firstIndex = firstIndex, .indexCount = data.indexCount, .firstVertex = 0};
  job_ranges(data.interpJob, whole.uniform, whole.verts, whole.pixel);
  bool asOne = true;
  if (interpolated && data.batchSize > 1 && data.interpJob != UINT32_MAX) {
    for (uint32_t i = 1; i < data.batchSize && asOne; ++i) {
      Range uniform, verts, pixel;
      job_ranges(data.interpJob + i, uniform, verts, pixel);
      const uint32_t first = batch_draw(data.batch + i).firstVertex;
      asOne = uniform.offset == whole.uniform.offset && uniform.size == whole.uniform.size &&
              pixel.offset == whole.pixel.offset && pixel.size == whole.pixel.size &&
              (whole.verts.size == 0 ? verts.size == 0
                                     : verts.size != 0 && verts.offset == whole.verts.offset +
                                                                              first * gxc::kVertexStrideBytes);
    }
    if (!asOne) {
      uint32_t index = firstIndex;
      for (uint32_t i = 0; i < data.batchSize; ++i) {
        const BatchDraw& part = batch_draw(data.batch + i);
        DrawPart& draw = parts.emplace_back(
            DrawPart{.firstIndex = index, .indexCount = part.indexCount, .firstVertex = part.firstVertex});
        job_ranges(data.interpJob + i, draw.uniform, draw.verts, draw.pixel);
        index += part.indexCount;
      }
    }
  }
  if (asOne)
    parts.push_back(whole);
}
} // namespace

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  const auto bind = [&](PipelineRef ref) {
    if (!bind_pipeline(ref, pass))
      return false;
    if (ref != g_pass.pipeline) {
      g_pass.pipeline = ref;
      g_pass.group1 = g_pass.group2 = g_pass.group3 = nullptr;
    }
    return true;
  };
  // Ensure both asynchronous pipelines are ready before the prepass writes any
  // depth. Binding the color pipeline first is harmless; it is rebound below.
  // A draw whose own are not ready yet is drawn with the ubershader, if it was
  // recorded with its data (submit_draw_plan), without the depth prepass.
  static const bool uberForced = ubershader_mode() == 2;
  bool uberDraw = false;
  if (g_softTarget) {
    // A run in the soft effects' buffer: their variant, checked ready before
    // the run was given the buffer.
    if (data.softPipeline == 0 || !bind(data.softPipeline)) {
      note_draw_left_out(data.batchSize);
      return;
    }
  } else if (uberForced && data.uberPipeline != 0) {
    if (!bind(data.uberPipeline)) {
      note_draw_left_out(data.batchSize);
      return;
    }
    uberDraw = true;
  } else if (!bind(data.pipeline) || (data.depthPipeline != 0 && !bind(data.depthPipeline))) {
    if (data.uberPipeline == 0 || !bind(data.uberPipeline)) {
      note_draw_left_out(data.batchSize);
      return;
    }
    uberDraw = true;
    note_ubershader_draw();
  }
  static std::vector<DrawPart> parts;
  collect_parts(data, parts);

  const auto setGroup1 = [&](const DrawPart& part) {
    const bool blended = part.uniform.size != 0;
    const auto& vsGroup = blended ? g_interpUniformBindGroup : g_uniformBindGroup;
    const uint32_t vsOffset = blended ? part.uniform.offset : data.uniformRange.offset;
    if (vsGroup.Get() != g_pass.group1 || vsOffset != g_pass.offset1) {
      pass.SetBindGroup(1, vsGroup, 1, &vsOffset);
      g_pass.group1 = vsGroup.Get();
      g_pass.offset1 = vsOffset;
    }
  };
  // The vertices: from the whole buffer at the draw's (or batch's) base
  // vertex, or a particle's blended ones from the in-between vertex area,
  // bound where the part's first index (counted from the batch's first
  // vertex) finds them.
  const auto setVertices = [&](const DrawPart& part) -> int32_t {
    const uint64_t back = uint64_t{part.firstVertex} * gxc::kVertexStrideBytes;
    if (part.verts.size != 0 && part.verts.offset >= back) {
      const uint64_t offset = part.verts.offset - back;
      if (g_pass.vertexBuffer != g_interpVertexBuffer.Get() || g_pass.vertexOffset != offset) {
        pass.SetVertexBuffer(0, g_interpVertexBuffer, offset);
        g_pass.vertexBuffer = g_interpVertexBuffer.Get();
        g_pass.vertexOffset = offset;
      }
      return 0;
    }
    if (data.vertRange.offset % gxc::kVertexStrideBytes == 0) {
      if (g_pass.vertexBuffer != g_vertexBuffer.Get() || g_pass.vertexOffset != 0) {
        pass.SetVertexBuffer(0, g_vertexBuffer);
        g_pass.vertexBuffer = g_vertexBuffer.Get();
        g_pass.vertexOffset = 0;
      }
      return static_cast<int32_t>(data.vertRange.offset / gxc::kVertexStrideBytes);
    }
    pass.SetVertexBuffer(0, g_vertexBuffer, data.vertRange.offset, data.vertRange.size);
    g_pass.vertexBuffer = g_vertexBuffer.Get();
    g_pass.vertexOffset = data.vertRange.offset;
    return 0;
  };
  if (!g_pass.indexBound) {
    pass.SetIndexBuffer(g_indexBuffer, wgpu::IndexFormat::Uint16);
    g_pass.indexBound = true;
  }
  // group 2 on the TEV path: the PS uniform (a dynamic-uniform bind group at
  // its own offset), or in an in-between frame its blended colours'.
  const auto setGroup2 = [&](const DrawPart& part) {
    const bool blended = part.pixel.size != 0;
    const auto& psGroup = blended ? g_interpUniformBindGroup : g_uniformBindGroup;
    const uint32_t psOffset = blended ? part.pixel.offset : data.pixelUniformRange.offset;
    if (psGroup.Get() != g_pass.group2 || psOffset != g_pass.offset2) {
      pass.SetBindGroup(2, psGroup, 1, &psOffset);
      g_pass.group2 = psGroup.Get();
      g_pass.offset2 = psOffset;
    }
  };
  const auto draw = [&](bool pixel) {
    for (const DrawPart& part : parts) {
      setGroup1(part);
      if (pixel)
        setGroup2(part);
      const int32_t baseVertex = setVertices(part);
      pass.DrawIndexed(part.indexCount, 1, part.firstIndex, baseVertex);
    }
  };
  if (uberDraw) {
    // group 2: the pixel constants and the shader key; group 3: all eight texmaps.
    const uint32_t psOffset = data.uberPixelRange.offset;
    if (g_uniformBindGroup.Get() != g_pass.group2 || psOffset != g_pass.offset2) {
      pass.SetBindGroup(2, g_uniformBindGroup, 1, &psOffset);
      g_pass.group2 = g_uniformBindGroup.Get();
      g_pass.offset2 = psOffset;
    }
    const auto& group = find_bind_group(data.uberTextureBindGroup);
    if (group.Get() != g_pass.group3) {
      pass.SetBindGroup(3, group);
      g_pass.group3 = group.Get();
    }
    draw(false);
    return;
  }
  if (data.depthPipeline != 0) {
    draw(false);
    if (!bind(data.pipeline))
      return;
  }
  if (data.tev) {
    // group 3 = texture (group 2, the PS uniform, is set for each part).
    if (data.textureBindGroup != 0) {
      const auto& group = find_bind_group(data.textureBindGroup);
      if (group.Get() != g_pass.group3) {
        pass.SetBindGroup(3, group);
        g_pass.group3 = group.Get();
      }
    }
  } else if (data.textureBindGroup != 0) {
    const auto& group = find_bind_group(data.textureBindGroup);
    if (group.Get() != g_pass.group2 || g_pass.offset2 != UINT32_MAX) {
      pass.SetBindGroup(2, group);
      g_pass.group2 = group.Get();
      g_pass.offset2 = UINT32_MAX;
    }
  }
  if (data.actorLight || data.pointLight) {
    // Wind Waker HD's character light (and its light buffer on the lit scenery): the pass's group 0 with its uniform,
    // shadow and lights, and the static group back
    // after (the other draws' pipelines, and the other draw types', expect it).
    pass.SetBindGroup(0, actor_light_pass_group());
    draw(data.tev);
    pass.SetBindGroup(0, g_staticBindGroup);
    return;
  }
  draw(data.tev);
}

// The sun shadow map's casters (post_fx.hpp). Every draw has the one decoded
// vertex layout and keeps its transforms in its constants, so two pipelines
// draw them all: one for positions through the current matrix, one through the
// per-vertex ones (a skinned model). Each takes the position to the camera's
// view space exactly as the draw's own shader does (gxcore_shader.cpp), then
// through `rows` to the map. Depth only, no fragment shader: nothing is
// compiled per draw, so no draw is left out of the map while it compiles.
namespace {
constexpr const char* kSunCasterShader = R"""(
struct Caster {
    rows: array<vec4f, 3>,
};
struct VertexShaderConstants {
    posnormalmatrix: array<vec4f, 6>,
    projection: array<vec4f, 4>,
    texmatrices: array<vec4f, 24>,
    transformmatrices: array<vec4f, 64>,
};
@group(0) @binding(0) var<uniform> caster: Caster;
@group(1) @binding(0) var<uniform> vsc: VertexShaderConstants;
struct VertexIn {
    @location(0) rawpos: vec3f,
    @location(1) posmtx: u32,
};
fn to_map(p0: vec4f, p1: vec4f, p2: vec4f, raw: vec3f) -> vec4f {
    let pos4 = vec4f(raw, 1.0);
    let viewpos = vec4f(dot(p0, pos4), dot(p1, pos4), dot(p2, pos4), 1.0);
    return vec4f(dot(caster.rows[0], viewpos), dot(caster.rows[1], viewpos), dot(caster.rows[2], viewpos), 1.0);
}
@vertex
fn vs_current(in: VertexIn) -> @builtin(position) vec4f {
    return to_map(vsc.posnormalmatrix[0], vsc.posnormalmatrix[1], vsc.posnormalmatrix[2], in.rawpos);
}
@vertex
fn vs_indexed(in: VertexIn) -> @builtin(position) vec4f {
    let i = i32(in.posmtx);
    return to_map(vsc.transformmatrices[i], vsc.transformmatrices[i + 1], vsc.transformmatrices[i + 2], in.rawpos);
}
// An alpha-tested caster: the first raw texture coordinate rides along, and
// the fragment keeps the texture's cutout (most such materials sample their
// texture with the identity, so the raw coordinate is the sample's).
struct VertexInUv {
    @location(0) rawpos: vec3f,
    @location(1) posmtx: u32,
    @location(2) uv0: vec2f,
};
struct VertexOutUv {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};
@vertex
fn vs_current_uv(in: VertexInUv) -> VertexOutUv {
    var out: VertexOutUv;
    out.pos = to_map(vsc.posnormalmatrix[0], vsc.posnormalmatrix[1], vsc.posnormalmatrix[2], in.rawpos);
    out.uv = in.uv0;
    return out;
}
@vertex
fn vs_indexed_uv(in: VertexInUv) -> VertexOutUv {
    let i = i32(in.posmtx);
    var out: VertexOutUv;
    out.pos = to_map(vsc.transformmatrices[i], vsc.transformmatrices[i + 1], vsc.transformmatrices[i + 2], in.rawpos);
    out.uv = in.uv0;
    return out;
}
@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;
@fragment
fn fs_alpha(in: VertexOutUv) {
    if (textureSample(tex, samp, in.uv).a < 0.5) {
        discard;
    }
}
)""";
constexpr uint64_t kSunCasterUniformSize = 12 * sizeof(float);
wgpu::BindGroupLayout g_sunCasterLayout;
// [character * 4 + alpha * 2 + indexed]: current, indexed, then the same with the alpha cutout; then the same for
// characters (BlueWake's own depth bias, see make_sun_caster_pipelines()).
// By [cull (none, back, front) x 8 + character x 4 + alpha x 2 + indexed].
std::array<wgpu::RenderPipeline, 24> g_sunCasterPipelines;
struct SunCasterState {
  int pipeline = -1;
  WGPUBindGroup group1 = nullptr;
  uint32_t offset1 = UINT32_MAX;
  WGPUBindGroup group2 = nullptr; // an alpha caster's texture
  WGPUBuffer vertexBuffer = nullptr;
  uint64_t vertexOffset = 0;
};
SunCasterState g_sunCasterState;

void make_sun_caster_pipelines() {
  wgpu::ShaderSourceWGSL source{};
  source.code = kSunCasterShader;
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Sun shadow map casters"};
  const auto module = g_device.CreateShaderModule(&moduleDescriptor);
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Vertex,
      .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = kSunCasterUniformSize},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "Sun shadow map casters", .entryCount = 1, .entries = &entry};
  g_sunCasterLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const std::array layouts{g_sunCasterLayout, g_uniformBindGroupLayout};
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "Sun shadow map casters", .bindGroupLayoutCount = layouts.size(), .bindGroupLayouts = layouts.data()};
  const auto pipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  const std::array alphaLayouts{g_sunCasterLayout, g_uniformBindGroupLayout, texture_bind_group_layout(1u)};
  const wgpu::PipelineLayoutDescriptor alphaLayoutDescriptor{.label = "Sun shadow map casters (alpha)",
                                                             .bindGroupLayoutCount = alphaLayouts.size(),
                                                             .bindGroupLayouts = alphaLayouts.data()};
  const auto alphaPipelineLayout = g_device.CreatePipelineLayout(&alphaLayoutDescriptor);
  const std::array attributes{
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x3, .offset = gxc::kVertexPosOffset, .shaderLocation = 0},
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Uint32, .offset = gxc::kVertexPosMtxOffset, .shaderLocation = 1},
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x2, .offset = gxc::kVertexUvOffset, .shaderLocation = 2},
  };
  const wgpu::VertexBufferLayout vertexLayout{
      .arrayStride = gxc::kVertexStrideBytes,
      .stepMode = wgpu::VertexStepMode::Vertex,
      .attributeCount = 2,
      .attributes = attributes.data(),
  };
  const wgpu::VertexBufferLayout alphaVertexLayout{
      .arrayStride = gxc::kVertexStrideBytes,
      .stepMode = wgpu::VertexStepMode::Vertex,
      .attributeCount = attributes.size(),
      .attributes = attributes.data(),
  };
  // Nearer the sun is less (the map is cleared to 1). The slope bias keeps a
  // surface steep to the sun from shadowing itself (post_fx adds a constant one);
  // at 2 texels it lifted the shadow off a thin caster's base, Link's shoes.
  // DOL_AURORA_SUN_SHADOW_MAP_CASTER_BIAS=constant,slope (local test; 0,1 by default): the casters' depth
  // biased as they are drawn, as Wind Waker HD draws its map (constant 80, slope 3, back faces culled: read from a
  // RenderDoc capture of Cemu; its map is 16-bit, so its constant is about 1 world unit in the first cascade), for
  // HD's own bias (post_fx DOL_AURORA_SUN_SHADOW_MAP_HD_BIAS).
  // Characters keep BlueWake's own (0, 1): with the two swapped (slope 80), Link's shadow came off his feet and his
  // arms thinned, and an island's cliff left a lit band on the water at its foot (Oct 31, Outset's bridge at 10:00).
  // DOL_AURORA_SUN_SHADOW_MAP_CASTER_BIAS_CHARACTERS=1 gives them HD's too (local test).
  int32_t casterBias = 0;
  float casterSlope = 1.f;
  if (const char* bias = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_CASTER_BIAS")) {
    casterBias = std::atoi(bias);
    if (const char* comma = std::strchr(bias, ','))
      casterSlope = static_cast<float>(std::atof(comma + 1));
  }
  const char* characterBias = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_CASTER_BIAS_CHARACTERS");
  const bool characterHdBias = characterBias != nullptr && characterBias[0] == '1';
  const wgpu::DepthStencilState depthStencil{
      .format = post_fx::kSunMapFormat,
      .depthWriteEnabled = true,
      .depthCompare = wgpu::CompareFunction::Less,
      .depthBias = casterBias,
      .depthBiasSlopeScale = casterSlope,
      .depthBiasClamp = 0.f,
  };
  const wgpu::DepthStencilState characterDepthStencil{
      .format = post_fx::kSunMapFormat,
      .depthWriteEnabled = true,
      .depthCompare = wgpu::CompareFunction::Less,
      .depthBias = characterHdBias ? casterBias : 0,
      .depthBiasSlopeScale = characterHdBias ? casterSlope : 1.f,
      .depthBiasClamp = 0.f,
  };
  const wgpu::FragmentState alphaFragment{.module = module, .entryPoint = "fs_alpha", .targetCount = 0};
  // DOL_AURORA_SUN_SHADOW_CHARACTER_CULL=front|back (local test; none by default): the skinned casters (Link,
  // the islanders) drawn into the map with one side culled. Wind Waker HD's Link doesn't shade his own side
  // turned from the sun, only what stands apart from it (his hair on his back), as a map of the casters' sides
  // turned away from the sun gives.
  auto characterCull = wgpu::CullMode::None;
  if (const char* cull = std::getenv("DOL_AURORA_SUN_SHADOW_CHARACTER_CULL"))
    characterCull = cull[0] == 'f' ? wgpu::CullMode::Front : cull[0] == 'b' ? wgpu::CullMode::Back : wgpu::CullMode::None;
  // Wind Waker HD culls its casters as their materials do (its map's draws in captures-nov3 s01, Dragon Roost
  // Cavern: 225 with back faces culled, 15 with none), so what faces away from the light (a closed room's ceiling,
  // seen from below) casts nothing; with HD's bias (80, 3) for the faces that do. The scenery's casters take the
  // draw's own cull mode while post_fx::sun_caster_cull() is on; characters keep characterCull.
  static constexpr wgpu::CullMode kCasterCull[3] = {wgpu::CullMode::None, wgpu::CullMode::Back,
                                                    wgpu::CullMode::Front};
  for (int cull = 0; cull < 3; ++cull) {
  for (int character = 0; character < 2; ++character) {
  for (int alpha = 0; alpha < 2; ++alpha) {
    for (int indexed = 0; indexed < 2; ++indexed) {
      const wgpu::RenderPipelineDescriptor descriptor{
          .label = alpha != 0 ? "Sun shadow map casters (alpha cutout)"
                 : indexed != 0 ? "Sun shadow map casters (per-vertex matrices)"
                                : "Sun shadow map casters",
          .layout = alpha != 0 ? alphaPipelineLayout : pipelineLayout,
          .vertex = {.module = module,
                     .entryPoint = alpha != 0 ? (indexed != 0 ? "vs_indexed_uv" : "vs_current_uv")
                                              : (indexed != 0 ? "vs_indexed" : "vs_current"),
                     .bufferCount = 1,
                     .buffers = alpha != 0 ? &alphaVertexLayout : &vertexLayout},
          .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList,
                        .frontFace = cull != 0 ? wgpu::FrontFace::CW : wgpu::FrontFace::CCW,
                        .cullMode = cull != 0 ? kCasterCull[cull]
                                    : indexed != 0 ? characterCull : wgpu::CullMode::None},
          .depthStencil = character != 0 ? &characterDepthStencil : &depthStencil,
          .multisample = {.count = 1, .mask = UINT32_MAX},
          .fragment = alpha != 0 ? &alphaFragment : nullptr,
      };
      g_sunCasterPipelines[cull * 8 + character * 4 + alpha * 2 + indexed] =
          g_device.CreateRenderPipeline(&descriptor);
    }
  }
  }
  }
}
} // namespace

void begin_sun_casters(const wgpu::RenderPassEncoder& pass, const float rows[12]) {
  if (!g_sunCasterPipelines[0])
    make_sun_caster_pipelines();
  // A small buffer for each pass, so a frame still in flight keeps its matrix.
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Sun shadow map casters",
      .usage = wgpu::BufferUsage::Uniform,
      .size = kSunCasterUniformSize,
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, kSunCasterUniformSize), rows, kSunCasterUniformSize);
  buffer.Unmap();
  const wgpu::BindGroupEntry entry{.binding = 0, .buffer = buffer, .offset = 0, .size = kSunCasterUniformSize};
  const wgpu::BindGroupDescriptor groupDescriptor{.layout = g_sunCasterLayout, .entryCount = 1, .entries = &entry};
  pass.SetBindGroup(0, g_device.CreateBindGroup(&groupDescriptor));
  pass.SetIndexBuffer(g_indexBuffer, wgpu::IndexFormat::Uint16);
  g_sunCasterState = SunCasterState{};
}

namespace {
// A draw's parts, as render() draws them: its own vertices and constants, or
// in an in-between frame the blended ones (the sun map's casters, the sea mask).
void draw_parts(const DrawData& data, const wgpu::RenderPassEncoder& pass, SunCasterState& state) {
  static std::vector<DrawPart> parts;
  collect_parts(data, parts);
  for (const DrawPart& part : parts) {
    const bool blended = part.uniform.size != 0;
    const auto& group = blended ? g_interpUniformBindGroup : g_uniformBindGroup;
    const uint32_t offset = blended ? part.uniform.offset : data.uniformRange.offset;
    if (group.Get() != state.group1 || offset != state.offset1) {
      pass.SetBindGroup(1, group, 1, &offset);
      state.group1 = group.Get();
      state.offset1 = offset;
    }
    // As render(): the whole vertex buffer at the draw's base vertex, or a
    // particle's blended vertices.
    int32_t baseVertex = 0;
    const uint64_t back = uint64_t{part.firstVertex} * gxc::kVertexStrideBytes;
    if (part.verts.size != 0 && part.verts.offset >= back) {
      const uint64_t vertexOffset = part.verts.offset - back;
      if (state.vertexBuffer != g_interpVertexBuffer.Get() || state.vertexOffset != vertexOffset) {
        pass.SetVertexBuffer(0, g_interpVertexBuffer, vertexOffset);
        state.vertexBuffer = g_interpVertexBuffer.Get();
        state.vertexOffset = vertexOffset;
      }
    } else if (data.vertRange.offset % gxc::kVertexStrideBytes == 0) {
      if (state.vertexBuffer != g_vertexBuffer.Get() || state.vertexOffset != 0) {
        pass.SetVertexBuffer(0, g_vertexBuffer);
        state.vertexBuffer = g_vertexBuffer.Get();
        state.vertexOffset = 0;
      }
      baseVertex = static_cast<int32_t>(data.vertRange.offset / gxc::kVertexStrideBytes);
    } else {
      pass.SetVertexBuffer(0, g_vertexBuffer, data.vertRange.offset, data.vertRange.size);
      state.vertexBuffer = g_vertexBuffer.Get();
      state.vertexOffset = data.vertRange.offset;
    }
    pass.DrawIndexed(part.indexCount, 1, part.firstIndex, baseVertex);
  }
}

} // namespace

void render_sun_caster(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  const bool alpha = data.sunCasterAlpha && data.cutoutBindGroup != 0;
  if (!data.sunCaster && !alpha)
    return;
  SunCasterState& state = g_sunCasterState;
  // Skinned scenery (skinned_scenery(), a palm's leaves) casts with the scenery's bias and HD's lookup (no slope
  // term), or its thin leaves shaded themselves.
  const bool character = data.character && (data.toon || !skinned_scenery());
  int cull = 0;
  if (!character && post_fx::sun_caster_cull()) {
    const auto mode = static_cast<gxc::CullMode>(data.sunCasterCull);
    cull = mode == gxc::CullMode::Back ? 1 : mode == gxc::CullMode::Front ? 2 : 0;
  }
  const int pipeline = cull * 8 + (character ? 4 : 0) + (alpha ? 2 : 0) + (data.posMtxIdx ? 1 : 0);
  if (state.pipeline != pipeline) {
    pass.SetPipeline(g_sunCasterPipelines[pipeline]);
    state.pipeline = pipeline;
  }
  if (alpha) {
    const auto& group = find_bind_group(data.cutoutBindGroup);
    if (group.Get() != state.group2) {
      pass.SetBindGroup(2, group);
      state.group2 = group.Get();
    }
  }
  draw_parts(data, pass, state);
}

namespace {
constexpr const char* kSeaMaskShader = R"""(
struct VertexShaderConstants {
    posnormalmatrix: array<vec4f, 6>,
    projection: array<vec4f, 4>,
    texmatrices: array<vec4f, 24>,
    transformmatrices: array<vec4f, 64>,
};
@group(1) @binding(0) var<uniform> vsc: VertexShaderConstants;
// Wind Waker HD's bloom mask (post_fx::bloom_mask_wanted()): blue 0.5 where what is drawn may glow (HD's
// offset 0), 0 elsewhere (-1); the pass clears it to 0.5, so what nothing covers (the sky) may glow.
override bloom_b: f32 = 0.0;
// Alpha 1: a character drawn with Wind Waker HD's light (DrawData::actorLight; the effects after the scene leave its
// pixels to it, post_fx.hpp actor_light_applied()); 0 everywhere else, which covers it.
override actor_a: f32 = 0.0;
// A character's green: 1 toon-shaded (Link, the islanders: Wind Waker HD's rim light takes those), 0.6 skinned
// only (a palm's swaying leaves); both are characters to the sun shadows.
override actor_g: f32 = 1.0;
struct VertexIn {
    @location(0) rawpos: vec3f,
    @location(1) posmtx: u32,
};
// As the game's own vertex shader (gxcore_shader.cpp vs_main).
fn to_clip(p0: vec4f, p1: vec4f, p2: vec4f, raw: vec3f) -> vec4f {
    let pos4 = vec4f(raw, 1.0);
    let viewpos = vec4f(dot(p0, pos4), dot(p1, pos4), dot(p2, pos4), 1.0);
    var clip = vec4f(dot(vsc.projection[0], viewpos), dot(vsc.projection[1], viewpos),
                     dot(vsc.projection[2], viewpos), dot(vsc.projection[3], viewpos));
    clip.z = -clip.z;
    return clip;
}
@vertex
fn vs_current(in: VertexIn) -> @builtin(position) vec4f {
    return to_clip(vsc.posnormalmatrix[0], vsc.posnormalmatrix[1], vsc.posnormalmatrix[2], in.rawpos);
}
@vertex
fn vs_indexed(in: VertexIn) -> @builtin(position) vec4f {
    let i = i32(in.posmtx);
    return to_clip(vsc.transformmatrices[i], vsc.transformmatrices[i + 1], vsc.transformmatrices[i + 2], in.rawpos);
}
@fragment
fn fs_land() -> @location(0) vec4f {
    return vec4f(0.0, 0.0, bloom_b, 0.0);
}
@fragment
fn fs_sea() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 0.0);
}
// A character (a solid skinned draw), for the sun shadows' facing term.
@fragment
fn fs_actor() -> @location(0) vec4f {
    return vec4f(0.0, actor_g, 0.0, actor_a);
}
// What is not the sea and alpha-tested (a wave's crest, leaves): it covers the
// sea only where its texture is solid, as the sun map's cutout casters do.
struct VertexInUv {
    @location(0) rawpos: vec3f,
    @location(1) posmtx: u32,
    @location(2) uv0: vec2f,
};
struct VertexOutUv {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};
@vertex
fn vs_current_uv(in: VertexInUv) -> VertexOutUv {
    var out: VertexOutUv;
    out.pos = to_clip(vsc.posnormalmatrix[0], vsc.posnormalmatrix[1], vsc.posnormalmatrix[2], in.rawpos);
    out.uv = in.uv0;
    return out;
}
@vertex
fn vs_indexed_uv(in: VertexInUv) -> VertexOutUv {
    let i = i32(in.posmtx);
    var out: VertexOutUv;
    out.pos = to_clip(vsc.transformmatrices[i], vsc.transformmatrices[i + 1], vsc.transformmatrices[i + 2], in.rawpos);
    out.uv = in.uv0;
    return out;
}
@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;
@fragment
fn fs_land_alpha(in: VertexOutUv) -> @location(0) vec4f {
    if (textureSample(tex, samp, in.uv).a < 0.5) {
        discard;
    }
    return vec4f(0.0, 0.0, bloom_b, 0.0);
}
// A character's alpha-tested part (Link's hair), where its texture is solid.
@fragment
fn fs_actor_alpha(in: VertexOutUv) -> @location(0) vec4f {
    if (textureSample(tex, samp, in.uv).a < 0.5) {
        discard;
    }
    return vec4f(0.0, actor_g, 0.0, actor_a);
}

// With the scenery's smooth normals (the sun shadows' facing term): the same, and a second target, the
// view-space normal x 0.5 + 0.5 with alpha 1 where a draw gives one (0: the facing term takes the depth's).
struct MaskOut {
    @location(0) mask: vec4f,
    @location(1) normal: vec4f,
};
fn mask_only(mask: vec4f) -> MaskOut {
    return MaskOut(mask, vec4f(0.0));
}
@fragment
fn fs_land_n() -> MaskOut {
    return mask_only(vec4f(0.0, 0.0, bloom_b, 0.0));
}
@fragment
fn fs_sea_n() -> MaskOut {
    return mask_only(vec4f(1.0, 0.0, 0.0, 0.0));
}
@fragment
fn fs_actor_n() -> MaskOut {
    return mask_only(vec4f(0.0, actor_g, 0.0, actor_a));
}
@fragment
fn fs_land_alpha_n(in: VertexOutUv) -> MaskOut {
    if (textureSample(tex, samp, in.uv).a < 0.5) {
        discard;
    }
    return mask_only(vec4f(0.0, 0.0, bloom_b, 0.0));
}
@fragment
fn fs_actor_alpha_n(in: VertexOutUv) -> MaskOut {
    if (textureSample(tex, samp, in.uv).a < 0.5) {
        discard;
    }
    return mask_only(vec4f(0.0, actor_g, 0.0, actor_a));
}
// Scenery with smooth normals in its vertices' normal slot (made on the CPU, submit_draw_plan()): the normal
// turned into view space by the position matrix (the scenery's is a rotation, scale and move), blended across
// each triangle as HD's models' per-vertex light is.
// Colour 1's alpha below -0.5: a lava vertex (hd_lava_draw()), -(1 + the share of its light that is the lava's
// glow), which goes into the normal's alpha as 1 - 0.49 x the share for the sun pass to leave unshaded.
struct VertexInN {
    @location(0) rawpos: vec3f,
    @location(1) posmtx: u32,
    @location(3) normal: vec3f,
    @location(4) glow: vec4f,
};
struct VertexOutN {
    @builtin(position) pos: vec4f,
    @location(0) viewpos: vec3f,
    @location(1) normal: vec3f,
    @location(2) glow: f32,
};
fn smooth_out(p0: vec4f, p1: vec4f, p2: vec4f, in: VertexInN) -> VertexOutN {
    var out: VertexOutN;
    out.pos = to_clip(p0, p1, p2, in.rawpos);
    let pos4 = vec4f(in.rawpos, 1.0);
    out.viewpos = vec3f(dot(p0, pos4), dot(p1, pos4), dot(p2, pos4));
    // Unit at each vertex, as HD's VS normalizes its turned normal: blended across the triangle, its y is then HD's
    // per-vertex N.y blended (its hemisphere colour is per vertex, so linear across the triangle).
    let nv = vec3f(dot(p0.xyz, in.normal), dot(p1.xyz, in.normal), dot(p2.xyz, in.normal));
    out.normal = select(vec3f(0.0), nv / max(length(nv), 1e-12), length(nv) > 1e-6);
    out.glow = select(0.0, clamp(-in.glow.a - 1.0, 0.0, 1.0), in.glow.a < -0.5);
    return out;
}
@vertex
fn vs_current_n(in: VertexInN) -> VertexOutN {
    return smooth_out(vsc.posnormalmatrix[0], vsc.posnormalmatrix[1], vsc.posnormalmatrix[2], in);
}
@vertex
fn vs_indexed_n(in: VertexInN) -> VertexOutN {
    let i = i32(in.posmtx);
    return smooth_out(vsc.transformmatrices[i], vsc.transformmatrices[i + 1], vsc.transformmatrices[i + 2], in);
}
// The normal turned to the side the camera sees, by the triangle's own facing (a model's winding isn't known
// here); alpha 0 where the draw gives none. Left as blended, not renormalized (its users normalize it): the sun
// pass's hemisphere takes its y as is, HD's per-vertex N.y blended across the triangle.
fn seen_normal(in: VertexOutN) -> vec4f {
    var face = cross(dpdx(in.viewpos), dpdy(in.viewpos));
    if (dot(face, -in.viewpos) < 0.0) {
        face = -face;
    }
    var n = in.normal;
    let len = length(n);
    if (len < 1e-6) {
        return vec4f(0.0);
    }
    if (dot(n, face) < 0.0) {
        n = -n;
    }
    return vec4f(n * 0.5 + 0.5, 1.0);
}
@fragment
fn fs_land_smooth_n(in: VertexOutN) -> MaskOut {
    var n = seen_normal(in);
    if (n.a > 0.5 && in.glow > 0.0) {
        n.a = 1.0 - 0.49 * in.glow;
    }
    return MaskOut(vec4f(0.0, 0.0, bloom_b, 0.0), n);
}
// A character's cut-out part with its own normals (a palm's leaves): kind 5's cutout, kind 8's normal.
struct VertexInUvN {
    @location(0) rawpos: vec3f,
    @location(1) posmtx: u32,
    @location(2) uv0: vec2f,
    @location(3) normal: vec3f,
};
struct VertexOutUvN {
    @builtin(position) pos: vec4f,
    @location(0) viewpos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};
fn smooth_uv_out(p0: vec4f, p1: vec4f, p2: vec4f, in: VertexInUvN) -> VertexOutUvN {
    var out: VertexOutUvN;
    out.pos = to_clip(p0, p1, p2, in.rawpos);
    let pos4 = vec4f(in.rawpos, 1.0);
    out.viewpos = vec3f(dot(p0, pos4), dot(p1, pos4), dot(p2, pos4));
    out.normal = vec3f(dot(p0.xyz, in.normal), dot(p1.xyz, in.normal), dot(p2.xyz, in.normal));
    out.uv = in.uv0;
    return out;
}
@vertex
fn vs_current_uvn(in: VertexInUvN) -> VertexOutUvN {
    return smooth_uv_out(vsc.posnormalmatrix[0], vsc.posnormalmatrix[1], vsc.posnormalmatrix[2], in);
}
@vertex
fn vs_indexed_uvn(in: VertexInUvN) -> VertexOutUvN {
    let i = i32(in.posmtx);
    return smooth_uv_out(vsc.transformmatrices[i], vsc.transformmatrices[i + 1], vsc.transformmatrices[i + 2], in);
}
@fragment
fn fs_actor_alpha_smooth_n(in: VertexOutUvN) -> MaskOut {
    // The normal (its derivatives) and the texture before the discard, in uniform control flow.
    let n = seen_normal(VertexOutN(in.pos, in.viewpos, in.normal, 0.0));
    if (textureSample(tex, samp, in.uv).a < 0.5) {
        discard;
    }
    return MaskOut(vec4f(0.0, actor_g, 0.0, actor_a), n);
}
// A character with its own normals (Wind Waker HD's rim light): green in the mask, its normal beside it. Its
// position matrix turns the normal (a skinned part's is a rotation, as the scenery's is).
@fragment
fn fs_actor_smooth_n(in: VertexOutN) -> MaskOut {
    return MaskOut(vec4f(0.0, actor_g, 0.0, actor_a), seen_normal(in));
}
)""";

wgpu::BindGroupLayout g_seaMaskLayout; // empty: the draws' constants are group 1, as render() has them
wgpu::BindGroup g_seaMaskGroup;
// Made when first used (sea_mask_pipeline()), [((normals * 2 + bloom) * kinds + kind) * 2 + indexed]: kind 0 what
// is not the sea, 1 the sea, 2 a sea draw that writes no depth, 3 what is not the sea with its texture's cutout, 4 a
// character (skinned: green in the mask), 5 a character's part with its texture's cutout, 6 scenery with smooth
// normals (as 0 without the normals' target), 7 what is solid but writes no depth (the sky, for HD's bloom mask),
// 8 a character with its own normals (as 4 without the normals' target), 9 a character's cut-out part with its own
// normals (a palm's leaves; as 5 without the normals' target), 10 blended scenery that writes depth
// (DrawData::maskOccluder; as 0, but the sea's red under it is kept: a wave's crest stays the sea), 11 the same with
// its texture's cutout (as 3), 12 and 13 as 10 and 11 but lit by the game (DrawData::maskOccluderLit: a bush's
// leaves), which is not the sea, so the sea's red under it is cleared too; normals 1: the pass has the normals' second target; variant 1: blue 0.5 (HD's bloom mask: it may glow), or for a character green 0.6 (skinned, not
// toon-shaded). And [hd * the rest]: hd 1, a character drawn with Wind Waker HD's light (alpha 1 in the mask).
constexpr int kSeaMaskKinds = 14;
constexpr int kSeaMaskVariants = 2 * 2 * kSeaMaskKinds * 2;
std::array<wgpu::RenderPipeline, 2 * kSeaMaskVariants> g_seaMaskPipelines;
wgpu::ShaderModule g_seaMaskModule;
wgpu::PipelineLayout g_seaMaskPipelineLayout;
wgpu::PipelineLayout g_seaMaskAlphaPipelineLayout;
SunCasterState g_seaMaskState;
int g_seaMaskPipeline = -1;
bool g_seaMaskNormals = false;

void make_sea_mask_layouts() {
  wgpu::ShaderSourceWGSL source{};
  source.code = kSeaMaskShader;
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Sea mask"};
  g_seaMaskModule = g_device.CreateShaderModule(&moduleDescriptor);
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{.label = "Sea mask", .entryCount = 0, .entries = nullptr};
  g_seaMaskLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::BindGroupDescriptor groupDescriptor{.layout = g_seaMaskLayout, .entryCount = 0, .entries = nullptr};
  g_seaMaskGroup = g_device.CreateBindGroup(&groupDescriptor);
  const std::array layouts{g_seaMaskLayout, g_uniformBindGroupLayout};
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{
      .label = "Sea mask", .bindGroupLayoutCount = layouts.size(), .bindGroupLayouts = layouts.data()};
  g_seaMaskPipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  const std::array alphaLayouts{g_seaMaskLayout, g_uniformBindGroupLayout, texture_bind_group_layout(1u)};
  const wgpu::PipelineLayoutDescriptor alphaLayoutDescriptor{
      .label = "Sea mask (alpha)", .bindGroupLayoutCount = alphaLayouts.size(), .bindGroupLayouts = alphaLayouts.data()};
  g_seaMaskAlphaPipelineLayout = g_device.CreatePipelineLayout(&alphaLayoutDescriptor);
}

const wgpu::RenderPipeline& sea_mask_pipeline(int normals, int bloom, int kind, int indexed, int hd) {
  auto& pipeline =
      g_seaMaskPipelines[hd * kSeaMaskVariants + ((normals * 2 + bloom) * kSeaMaskKinds + kind) * 2 + indexed];
  if (pipeline)
    return pipeline;
  if (!g_seaMaskModule)
    make_sea_mask_layouts();
  const std::array attributes{
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x3, .offset = gxc::kVertexPosOffset, .shaderLocation = 0},
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Uint32, .offset = gxc::kVertexPosMtxOffset, .shaderLocation = 1},
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x2, .offset = gxc::kVertexUvOffset, .shaderLocation = 2},
  };
  const std::array normalAttributes{
      attributes[0],
      attributes[1],
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x3, .offset = gxc::kVertexNormalOffset, .shaderLocation = 3},
      wgpu::VertexAttribute{.format = wgpu::VertexFormat::Float32x4, .offset = gxc::kVertexColor1Offset, .shaderLocation = 4},
  };
  const wgpu::VertexBufferLayout normalVertexLayout{
      .arrayStride = gxc::kVertexStrideBytes,
      .stepMode = wgpu::VertexStepMode::Vertex,
      .attributeCount = normalAttributes.size(),
      .attributes = normalAttributes.data(),
  };
  const std::array uvNormalAttributes{attributes[0], attributes[1], attributes[2], normalAttributes[2]};
  const wgpu::VertexBufferLayout uvNormalVertexLayout{
      .arrayStride = gxc::kVertexStrideBytes,
      .stepMode = wgpu::VertexStepMode::Vertex,
      .attributeCount = uvNormalAttributes.size(),
      .attributes = uvNormalAttributes.data(),
  };
  const wgpu::VertexBufferLayout vertexLayout{
      .arrayStride = gxc::kVertexStrideBytes,
      .stepMode = wgpu::VertexStepMode::Vertex,
      .attributeCount = 2,
      .attributes = attributes.data(),
  };
  const wgpu::VertexBufferLayout alphaVertexLayout{
      .arrayStride = gxc::kVertexStrideBytes,
      .stepMode = wgpu::VertexStepMode::Vertex,
      .attributeCount = attributes.size(),
      .attributes = attributes.data(),
  };
  // An unlit blended occluder (kinds 10 and 11: the waves' crest strips, the horizon's wave sheets) keeps the sea's
  // red under it, so it stays the sea. A lit one (12 and 13: a bush's leaf cards) is not the sea and clears it:
  // kept, the sea behind the bush marked the bush's pixels as the sea, with the shapes of what stood in front of
  // the sea (the King of Red Lions, the dock, Beedle's shop) cut out of it.
  const bool seaKept = kind == 10 || kind == 11;
  const std::array targets{
      wgpu::ColorTargetState{.format = post_fx::kSeaMaskFormat,
                             // Alpha too: every draw writes 0 but a Wind Waker HD-lit character's 1 (actor_a).
                             .writeMask = seaKept ? wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue |
                                                        wgpu::ColorWriteMask::Alpha
                                                  : wgpu::ColorWriteMask::All},
      wgpu::ColorTargetState{.format = post_fx::kSunNormalFormat},
  };
  // Reversed Z: nearer is more; level surfaces drawn later win, as with the game's less-or-equal.
  const wgpu::DepthStencilState depth{
      .format = post_fx::kSeaMaskDepthFormat,
      .depthWriteEnabled = kind != 2 && kind != 7,
      .depthCompare = wgpu::CompareFunction::GreaterEqual,
  };
  const bool alpha = kind == 3 || kind == 5 || kind == 9 || kind == 11 || kind == 13;
  const bool smooth = (kind == 6 || kind == 8) && normals != 0;
  const bool alphaSmooth = kind == 9 && normals != 0;
  const char* entry = kind == 5 || kind == 9              ? "fs_actor_alpha"
                      : alpha                               ? "fs_land_alpha"
                      : kind == 0 || kind == 6 || kind == 7 || kind == 10 || kind == 12 ? "fs_land"
                      : kind == 4 || kind == 8              ? "fs_actor"
                                                            : "fs_sea";
  const std::string entryN = alphaSmooth ? "fs_actor_alpha_smooth_n"
                             : smooth    ? (kind == 8 ? "fs_actor_smooth_n" : "fs_land_smooth_n")
                                         : std::string(entry) + "_n";
  const std::array constants{wgpu::ConstantEntry{.key = "bloom_b", .value = bloom != 0 ? 0.5 : 0.0},
                             wgpu::ConstantEntry{.key = "actor_g", .value = bloom != 0 ? 0.6 : 1.0},
                             wgpu::ConstantEntry{.key = "actor_a", .value = hd != 0 ? 1.0 : 0.0}};
  const wgpu::FragmentState fragment{.module = g_seaMaskModule,
                                     .entryPoint = normals != 0 ? entryN.c_str() : entry,
                                     .constantCount = constants.size(),
                                     .constants = constants.data(),
                                     .targetCount = normals != 0 ? 2u : 1u,
                                     .targets = targets.data()};
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Sea mask",
      .layout = alpha ? g_seaMaskAlphaPipelineLayout : g_seaMaskPipelineLayout,
      .vertex = {.module = g_seaMaskModule,
                 .entryPoint = alphaSmooth ? (indexed != 0 ? "vs_indexed_uvn" : "vs_current_uvn")
                               : smooth  ? (indexed != 0 ? "vs_indexed_n" : "vs_current_n")
                               : alpha ? (indexed != 0 ? "vs_indexed_uv" : "vs_current_uv")
                                       : (indexed != 0 ? "vs_indexed" : "vs_current"),
                 .bufferCount = 1,
                 .buffers = alphaSmooth ? &uvNormalVertexLayout
                            : smooth    ? &normalVertexLayout
                            : alpha     ? &alphaVertexLayout
                                        : &vertexLayout},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList, .cullMode = wgpu::CullMode::None},
      .depthStencil = &depth,
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  pipeline = g_device.CreateRenderPipeline(&descriptor);
  return pipeline;
}

} // namespace

void begin_sea_mask(const wgpu::RenderPassEncoder& pass, bool normals) {
  if (!g_seaMaskModule)
    make_sea_mask_layouts();
  pass.SetBindGroup(0, g_seaMaskGroup);
  pass.SetIndexBuffer(g_indexBuffer, wgpu::IndexFormat::Uint16);
  g_seaMaskState = SunCasterState{};
  g_seaMaskPipeline = -1;
  g_seaMaskNormals = normals;
}

void render_sea_mask(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
  // What covers the sea is what the sun map takes as solid: a blended draw
  // doesn't, and an alpha-tested one only where its texture is solid (a wave's
  // crest whole covered a box of sea, which was then brightened).
  const bool cutout = !data.seaSurface && !data.sunCaster && data.cutoutBindGroup != 0 &&
                      (data.sunCasterAlpha || data.maskOccluderCutout);
  // A skinned draw that writes depth is a character (Link, the islanders), whether or not it casts in the sun
  // map (Link's own draws don't all): the sun shadows' facing term leaves it out.
  const bool character = !data.seaSurface && data.character && data.depthWrite;
  // With Wind Waker HD's bloom mask, what is solid but writes no depth (the sky's layers) is drawn too. The pass
  // clears the mask's blue to 0.5 (may glow); what writes no depth keeps it and every other draw here covers it with
  // 0, so the sky glows and nothing else does, as Cemu showed at Outset. (Whether a draw is lit says nothing here:
  // the GameCube lights its sky's layers and draws its islands' terrain unlit, its light baked in the vertices.)
  const bool bloomMask = post_fx::bloom_mask_wanted();
  const bool cover = bloomMask && !data.seaSurface && data.maskCover;
  // Blended scenery that writes depth (DrawData::maskOccluder) covers what is behind it, as solid scenery.
  const bool occluder = !data.seaSurface && data.maskOccluder;
  if (!data.seaSurface && !data.sunCaster && !cutout && !character && !cover && !occluder)
    return;
  const int kind = data.seaSurface ? (data.depthWrite ? 1 : 2)
                   : character     ? (cutout ? (data.hasNormal && g_seaMaskNormals ? 9 : 5)
                                             : data.hasNormal && g_seaMaskNormals ? 8 : 4)
                   : occluder      ? (cutout ? 11 : 10) + (data.maskOccluderLit ? 2 : 0)
                   : cutout        ? 3
                   : cover         ? 7
                   : data.smoothNormal && g_seaMaskNormals ? 6
                                                           : 0;
  const int bloom = character ? (data.toon ? 0 : 1) : kind == 7 ? 1 : 0;
  const int indexed = data.posMtxIdx ? 1 : 0;
  // A character drawn with Wind Waker HD's light (a matched toon draw): alpha 1, for the effects to leave to it.
  const int hd = character && data.actorLight ? 1 : 0;
  const int pipeline =
      hd * kSeaMaskVariants + (((g_seaMaskNormals ? 2 : 0) + bloom) * kSeaMaskKinds + kind) * 2 + indexed;
  if (g_seaMaskPipeline != pipeline) {
    pass.SetPipeline(sea_mask_pipeline(g_seaMaskNormals ? 1 : 0, bloom, kind, indexed, hd));
    g_seaMaskPipeline = pipeline;
  }
  if (cutout) {
    const auto& group = find_bind_group(data.cutoutBindGroup);
    if (group.Get() != g_seaMaskState.group2) {
      pass.SetBindGroup(2, group);
      g_seaMaskState.group2 = group.Get();
    }
  }
  draw_parts(data, pass, g_seaMaskState);
}

void note_frame_presented() {
  // The grey composite's copy back in place of its haze texture (copy_efb_to_texture()).
  for (auto it = g_monotoneHazeRestore.rbegin(); it != g_monotoneHazeRestore.rend(); ++it) {
    const auto found = g_efbCopyTextures.find(it->address);
    if (found != g_efbCopyTextures.end() && found->second.handle == it->haze)
      found->second = it->copy;
  }
  g_monotoneHazeRestore.clear();
  g_textureVerifyFrame.fetch_add(1u, std::memory_order_relaxed);
  frame_interp::end_game_frame();
}

void reset_texture_cache() {
  g_textureCache.clear();
  g_textureAwaitingReplacement.clear();
  g_textureAddrKey.clear();
  g_efbCopyCache.clear();
  g_efbCopyTextures.clear();
  g_monotoneHazeTextures.clear();
  g_monotoneHazeRestore.clear();
  g_textureCacheStats = {};
}

void stop_interp_helper();

void shutdown() {
  // Stop the detached interpolation helper before it outlives the process:
  // it waits on `work` forever, so without a stop flag glibc's exit() teardown
  // hangs on the still-live detached thread. (Defined with InterpHelper.)
  stop_interp_helper();
  // Release every wgpu object we still hold before webgpu::shutdown() destroys
  // the device, the Vulkan instance and the XCB connection. The flat_hash_map
  // and the device/texture/view statics hold the last references; destroying
  // them at exit() (after the connection was freed) aborted in
  // xcb_send_request -> realloc with "double free or corruption (!prev)".
  {
    std::lock_guard lock{g_texBindLayoutMutex};
    g_texBindLayoutCache.clear();
    g_texBindLayoutOwner = {};
  }
  {
    std::lock_guard lock{g_emptyTexmapMutex};
    g_emptyTexmapView = {};
    g_emptyTexmapTexture = {};
    g_emptyTexmapOwner = {};
  }
  reset_texture_cache();
  g_sunCasterPipelines = {};
  g_sunCasterLayout = {};
  g_seaMaskPipelines = {};
  g_seaMaskPipelineLayout = {};
  g_seaMaskAlphaPipelineLayout = {};
  g_seaMaskModule = {};
  g_seaMaskGroup = {};
  g_seaMaskLayout = {};
  release_actor_light();
  {
    std::lock_guard lock{g_texBindLayoutMutex};
    g_actorLightLayout = {};
    g_actorLightLayoutOwner = {};
  }
}

void set_texture_dirty_epoch_observer(TextureDirtyEpochObserver observer) {
  g_textureDirtyEpochObserver = observer;
}

void set_efb_copy_epoch_observer(TextureDirtyEpochObserver observer) { g_efbCopyEpochObserver = observer; }

const TextureCacheStats& texture_cache_stats() { return g_textureCacheStats; }
unsigned long long texture_upload_count() { return g_textureCacheStats.uploads; }

// Resolve the current EFB region into a texture the guest-identity path can bind
// later, keyed by the copy's destination address (63/S16). Mirrors aurora
// lib/dolphin/gx/GXFrameBuffer.cpp copy_tex on the same substrate: map the EFB
// source rect through the logical->render scaling, allocate (or reuse) a resolve
// target, then gfx::resolve_pass — which converts per format (tex_copy_conv,
// incl. depth targets) and optionally clears the EFB to the game's copy-clear
// color/Z afterwards. The bound draw's geometry is already in the pass (the
// sink flushed the pending draw before firing this).
std::vector<uint8_t> read_efb_copy(const gxc::EfbCopyCommand& cmd) {
  const auto it = g_efbCopyTextures.find(cmd.dest_address);
  if (it == g_efbCopyTextures.end() || !it->second.handle ||
      (cmd.format != 1u && cmd.format != 4u))
    return {};
  const auto texture = it->second.handle;
  // Submission keeps the current EFB and does not present an extra frame.
  // The following worker readback is ordered after the copy's render pass.
  if (!gfx::segment_frame())
    return {};
  const auto rgba = gfx::read_texture_rgba8(texture);
  return gxc::encode_efb_copy(cmd.format, cmd.destination_width,
      cmd.destination_height, rgba, texture->size.width, texture->size.height);
}

// The grey composite's frame copy while monotone (copy_efb_to_texture()), whose quad is left out; 0: none.
static uint32_t g_monotoneDrop = 0;
// The format of the frame's last EFB copy.
static uint32_t g_lastCopyFormat = 0;

void copy_efb_to_texture(const gxc::EfbCopyCommand& cmd) {
  // Mirror the copy-clear color/Z (BP 0x4F-0x51 at this copy) into the gx
  // state GXSetCopyClear would have written: begin_frame's pass-0 EFB clear
  // reads g_gxState.clearColor/clearDepth, and in gxcore mode the live gx
  // layer that normally maintains them is bypassed. Without this every frame
  // cleared to the default alpha=1 and e.g. the Strikers shadow-grab alpha
  // background inverted (glxSwap display-copy clears to {0,0,0,0}).
  gx::g_gxState.clearColor = {
      static_cast<float>(cmd.clear_r) / 255.f,
      static_cast<float>(cmd.clear_g) / 255.f,
      static_cast<float>(cmd.clear_b) / 255.f,
      static_cast<float>(cmd.clear_a) / 255.f,
  };
  gx::g_gxState.clearDepth = cmd.clear_z;
  if (cmd.format == 0xFu) {
    // Display copy (GXCopyDisp): no texture destination; its requested clear
    // becomes the next frame's EFB clear via the state mirrored above.
    g_monotoneDrop = 0;
    g_lastCopyFormat = 0;
    return;
  }
  const auto fmt = static_cast<GXTexFmt>(cmd.format);
  const gfx::ClipRect srcRect = gx::map_logical_scissor(gx::copy_source(
      gfx::ClipRect{
          .x = static_cast<int32_t>(cmd.src_x),
          .y = static_cast<int32_t>(cmd.src_y),
          .width = static_cast<int32_t>(cmd.width),
          .height = static_cast<int32_t>(cmd.height),
      },
      cmd.destination_width, cmd.destination_height));
  // The copy is made at the render target's scale, as Aurora's own
  // GXCopyTex does (dolphin/gx/GXFrameBuffer.cpp scale_copy_dst) and as
  // Dolphin's scaled EFB copies do. Allocated at the guest's size, a game
  // that copies the whole EFB and draws it back (Wind Waker's depth of field
  // and blur passes, every frame) replaced its high-resolution scene with a
  // 640x480 image stretched over it. Sampling uses normalized coordinates,
  // so a larger texture needs nothing else.
  uint32_t dstWidth = std::max(cmd.destination_width, static_cast<uint32_t>(1));
  uint32_t dstHeight = std::max(cmd.destination_height, static_cast<uint32_t>(1));
  static const bool scale_copies = [] {
    const char* env = std::getenv("DOL_GXCORE_COPY_SCALE");
    return env == nullptr || env[0] != '0';
  }();
  if (scale_copies && gx::g_gxState.viewportPolicy != AURORA_VIEWPORT_NATIVE) {
    const auto [logicalW, logicalH] = gx::logical_fb_size();
    const auto [targetW, targetH] = gfx::get_render_target_size();
    if (logicalW != 0 && logicalH != 0 && targetW != 0 && targetH != 0) {
      const float sx = static_cast<float>(targetW) / static_cast<float>(logicalW);
      const float sy = static_cast<float>(targetH) / static_cast<float>(logicalH);
      dstWidth = std::max<uint32_t>(static_cast<uint32_t>(std::lround(dstWidth * sx)), 1u);
      dstHeight = std::max<uint32_t>(static_cast<uint32_t>(std::lround(dstHeight * sy)), 1u);
    }
  }

  const EfbCopyKey key{cmd.dest_address, dstWidth, dstHeight, cmd.format, !cmd.efb_has_alpha};
  auto it = g_efbCopyCache.find(key);
  if (it == g_efbCopyCache.end() || !it->second) {
    TextureHandle handle;
    if (gfx::tex_copy_conv::needs_conversion(fmt)) {
      handle = gfx::new_conv_texture(dstWidth, dstHeight, fmt, "GXCore Copy Conv");
    } else {
      // As Aurora's GXCopyTex: an RGB565 target, or an EFB without alpha,
      // gets a view whose alpha reads one.
      const auto viewFmt = fmt == GX_TF_RGB565 || !cmd.efb_has_alpha ? GX_TF_RGB565 : GX_TF_RGBA8;
      handle = gfx::new_render_texture(dstWidth, dstHeight, viewFmt, "GXCore Copy");
    }
    it = g_efbCopyCache.insert_or_assign(key, handle).first;
  }
  if (!it->second) {
    return;
  }
  uint64_t memoryEpoch = 0;
  const bool memoryEpochValid =
      cmd.byte_size != 0u && efb_copy_epoch(cmd.dest_address, cmd.byte_size, &memoryEpoch);
  g_efbCopyTextures.insert_or_assign(
      cmd.dest_address,
      EfbCopyBinding{it->second, cmd.byte_size, memoryEpoch, memoryEpochValid});
  // Wind Waker HD's monotone (post_fx::Monotone): Wind Waker's grey composite (m_Do_graphic.cpp drawDepth) copies
  // the depth (asked for as Z16; the copy's register reads Z24X8), then the frame as I8 to mDoGph_gInf_c::mFrameBufferTex while monotone, and lays that copy back
  // over the scene in one flat quad, before the player's lists. That copy marks the point (the scene's effects and
  // HD's monotone go in after it) and the quad is left out (submit_draw_plan()). A copy at an offset in it (the
  // frame's y below 0) is the same copy.
  g_monotoneDrop = 0;
  {
    const auto monotone = post_fx::monotone_params();
    const uint32_t dest = cmd.dest_address & 0x3FFFFFFFu;
    if (monotone.on && monotone.frameCopy != 0 && fmt == GX_TF_I8 && (g_lastCopyFormat & _GX_TF_ZTF) != 0u &&
        dest >= monotone.frameCopy && dest < monotone.frameCopy + 0x100000u) {
      // What samples the copy after it this frame (the heat haze, drawn after HD's monotone) takes a second texture
      // the copy is made into again from the frame with the monotone in (common.cpp render()); the next frame's
      // motion blur, laid back under the monotone, the copy itself (restored at the frame's end).
      auto& haze = g_monotoneHazeTextures[key];
      if (!haze)
        haze = gfx::new_conv_texture(dstWidth, dstHeight, fmt, "GXCore Copy Conv (monotone haze)");
      gfx::mark_monotone_copy(monotone, haze);
      if (haze) {
        auto& binding = g_efbCopyTextures[cmd.dest_address];
        g_monotoneHazeRestore.push_back(MonotoneHazeRestore{cmd.dest_address, binding, haze});
        binding.handle = haze;
      }
      g_monotoneDrop = dest;
      static bool s_logged = false;
      if (!s_logged) {
        s_logged = true;
        std::fprintf(stderr, "[monotone] the grey composite's copy (I8 to %08X): Wind Waker HD's monotone from here\n",
                     dest);
      }
    }
  }
  g_lastCopyFormat = cmd.format;
  float clearDepthValue = static_cast<float>(cmd.clear_z) / 16777215.f;
  if (gx::UseReversedZ) {
    clearDepthValue = 1.f - clearDepthValue;
  }
  gfx::resolve_pass(it->second, srcRect, cmd.clear && cmd.color_update,
                    cmd.clear && cmd.alpha_update,
                    cmd.clear && cmd.depth_update,
                    Vec4<float>{static_cast<float>(cmd.clear_r) / 255.f,
                                static_cast<float>(cmd.clear_g) / 255.f,
                                static_cast<float>(cmd.clear_b) / 255.f,
                                static_cast<float>(cmd.clear_a) / 255.f},
                    clearDepthValue, fmt);
}

// Uniform de-duplication across consecutive draws.
//
// Each draw carries the full vertex-constant block (every position, texture
// and normal matrix, about 2 KB) and a pixel-constant block, and consecutive
// draws of one model often carry byte-identical blocks. Pushing them anyway
// filled Aurora's 24 MB uniform staging area in the middle of Wind Waker's
// Outset frames, and each split waited on the GPU. A draw whose block matches
// the previous draw's, within the same frame packet, reuses that range.
// On Apple GPUs the block a cache last took is compared where it was staged
// (uniform_bytes, or interp_uniform_bytes for the in-between frames'): the
// staging area is ordinary memory there, and a copy of each block kept aside
// for the comparison was a tenth of the FIFO worker's copying. Elsewhere
// (Direct3D 12, Vulkan) the staging area is the GPU's upload memory, which the
// CPU reads uncached: comparing there took Wind Waker's Outset frames (14,000
// draws) 400 ms on the FIFO worker, so the block is kept aside.
#if defined(__APPLE__)
#define AURORA_UNIFORM_COMPARE_STAGED 1
#else
#define AURORA_UNIFORM_COMPARE_STAGED 0
#endif
struct UniformCache {
  Range range{};
  uint64_t frameId = 0;
  uint64_t hits = 0;
  uint64_t pushes = 0;
#if !AURORA_UNIFORM_COMPARE_STAGED
  std::vector<uint8_t> bytes; // the block range holds
#endif
};
static UniformCache g_vertexUniformCache;
// The constants_id of the plan whose constants g_vertexUniformCache holds the bytes of (0: none).
static uint64_t g_pushedConstantsId = 0;
static UniformCache g_interpUniformCache;
static UniformCache g_pixelUniformCache;
static UniformCache g_uberPixelUniformCache;

// The ubershader's texture for a texmap slot no TEV stage samples: one white
// texel, made once per device (the ubershader's bind group has all eight).
static const wgpu::TextureView& empty_texmap_view() {
  std::lock_guard lock{g_emptyTexmapMutex};
  if (g_emptyTexmapOwner.Get() != g_device.Get()) {
    g_emptyTexmapOwner = g_device;
    const wgpu::TextureDescriptor descriptor{
        .label = "GXCore ubershader empty texmap",
        .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {1, 1, 1},
        .format = wgpu::TextureFormat::RGBA8Unorm,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    g_emptyTexmapTexture = g_device.CreateTexture(&descriptor);
    g_emptyTexmapView = g_emptyTexmapTexture.CreateView();
    static constexpr uint8_t kWhite[4] = {255, 255, 255, 255};
    const wgpu::TexelCopyTextureInfo dst{.texture = g_emptyTexmapTexture};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = 4, .rowsPerImage = 1};
    const wgpu::Extent3D size{1, 1, 1};
    webgpu::g_queue.WriteTexture(&dst, kWhite, sizeof kWhite, &layout, &size);
  }
  return g_emptyTexmapView;
}

// The block the cache last took: `staged` where it was staged, or the copy.
static const uint8_t* cached_bytes(const UniformCache& cache, const uint8_t* staged) {
#if AURORA_UNIFORM_COMPARE_STAGED
  return staged;
#else
  (void)staged;
  return cache.bytes.data();
#endif
}

// The cache took `data` (staged at cache.range).
static void keep_cached(UniformCache& cache, const uint8_t* data, size_t length) {
#if AURORA_UNIFORM_COMPARE_STAGED
  (void)cache;
  (void)data;
  (void)length;
#else
  cache.bytes.assign(data, data + length);
#endif
}

// Whether `data` is the block the cache last took, in this frame packet.
static bool repeats_cached(const UniformCache& cache, const uint8_t* data, size_t length) {
  const uint64_t frameId = current_frame_id();
  return frameId != 0 && cache.frameId == frameId && cache.range.size == length &&
         std::memcmp(cached_bytes(cache, AURORA_UNIFORM_COMPARE_STAGED ? uniform_bytes(cache.range) : nullptr), data,
                     length) == 0;
}

// The in-between frame's blocks, de-duplicated the same way. repeated: the
// in-between frame made this block the way
// it made the last one, from the same constants (frame_interp::last_blend_repeated),
// so its bytes are the same.
static Range push_interp_uniform_dedup(UniformCache& cache, uint64_t frameId, size_t slot, const uint8_t* data,
                                       size_t length, bool repeated) {
  if (frameId != 0 && cache.frameId == frameId && cache.range.size == length &&
      (repeated ||
       std::memcmp(cached_bytes(cache, AURORA_UNIFORM_COMPARE_STAGED ? interp_uniform_bytes(slot, cache.range)
                                                                     : nullptr),
                   data, length) == 0)) {
    ++cache.hits;
    return cache.range;
  }
  ++cache.pushes;
  cache.range = push_interp_uniform(slot, data, length);
  cache.frameId = frameId;
  keep_cached(cache, data, length);
  return cache.range;
}

// --- In-between frames on a helper thread -----------------------------------
//
// Matching each draw to its counterpart in the frame before and blending its
// constants (frame_interp::blend_draw), then staging the blended block, was a
// fifth of the FIFO worker's time. That worker is what holds the game below
// 30 FPS in the busiest scenes (Outset's village, 10,000 draws a frame, on a
// Windows PC: 2026-09-29), so a helper thread does it, in draw order, while
// the worker goes on. Each draw records its job; end_frame waits for the
// helper before a frame packet is handed on, and the render worker reads the
// finished ranges (interp_job_ranges). One recording thread at a time queues
// jobs (the recording lock serialises them), so the queue is single-producer.
namespace {
struct InterpJob {
  frame_interp::DrawInput input; // its key, samples and a particle's positions
  std::vector<float> vertices;   // a particle's decoded vertices, for its blended copy
  uint64_t frameId;
  size_t slot;
  bool repeatsLastDraw; // the constants are the draw before's (repeats_cached)
  // The constants, copied only where they differ from the job before's
  // (constantsCopied); a job that repeats them leaves the 2.8 KB copy to the
  // helper, which keeps the last it was given (InterpHelper::constants).
  bool constantsCopied;
  gxc::VertexShaderConstants constants;
  // A TEV draw's pixel constants, the same way: copied only where they differ
  // from those the job before them carried (pixelCopied).
  bool tev;
  bool pixelCopied;
  gxc::PixelShaderConstants pixel;
};

// A draw's in-between vertices: replace its blended positions and direct UVs
// while preserving all other attributes in the current vertex stream.
Range push_blended_vertices(size_t slot, const std::vector<float>& vertices, const float* positions,
                            const float* texcoords, uint8_t texcoordMask, std::vector<float>& scratch) {
  scratch.assign(vertices.begin(), vertices.end());
  const size_t count = scratch.size() / gxc::kVertexFloats;
  size_t uvAt = 0;
  for (size_t i = 0; i < count; ++i) {
    float* vertex = scratch.data() + i * gxc::kVertexFloats;
    if (positions != nullptr)
      std::memcpy(vertex + gxc::kVertexPosOffset / sizeof(float), positions + i * 3u, sizeof(float) * 3u);
    if (texcoords != nullptr)
      for (unsigned uv = 0; uv < gxc::kMaxRawTexCoords; ++uv)
        if ((texcoordMask >> uv) & 1u) {
          std::memcpy(vertex + gxc::kVertexUvOffset / sizeof(float) + uv * 2u,
                      texcoords + uvAt, sizeof(float) * 2u);
          uvAt += 2;
        }
  }
  return push_interp_vertices(slot, reinterpret_cast<const uint8_t*>(scratch.data()), scratch.size() * sizeof(float));
}

struct InterpHelper {
  static constexpr uint64_t Capacity = 1024;
  // A sleeping helper is woken for this many jobs at once (or by
  // wait_interp_jobs at the frame's end), not for every draw.
  static constexpr uint64_t WakeBatch = 32;
  std::unique_ptr<InterpJob[]> ring{new InterpJob[Capacity]};
  std::atomic<uint64_t> produced{0};
  std::atomic<uint64_t> consumed{0};
  std::atomic<bool> sleeping{false};
  std::atomic<bool> stop{false};
  std::mutex mutex;
  std::condition_variable work;
  std::condition_variable idle;
  // Recording thread: the frame packet being queued and its job count, and
  // whether a job of it has carried pixel constants.
  uint64_t packet = 0;
  uint32_t jobs = 0;
  bool pixelCarried = false;
  // Helper thread: the last block it staged per step, and a particle's vertices.
  UniformCache cache[frame_interp::kMaxSteps];
  UniformCache pixelCache[frame_interp::kMaxSteps];
  std::vector<float> vertices;
  // Helper thread: the constants of the last job that carried them, which the
  // jobs after it that repeat them use; and the pixel constants likewise.
  gxc::VertexShaderConstants constants;
  gxc::PixelShaderConstants pixel;
};

// Made with the thread and never destroyed: the detached thread may still be
// waiting on it while statics are torn down at exit.
std::atomic<InterpHelper*> g_interpHelper{nullptr};

void interp_helper_main(InterpHelper* h) {
  thread_cpu::register_current(thread_cpu::Role::InterpHelper);
  for (;;) {
    const uint64_t tail = h->consumed.load(std::memory_order_relaxed);
    if (tail == h->produced.load(std::memory_order_seq_cst)) {
      // Draws come microseconds apart while a frame records: wait a little
      // before sleeping. Woken for every draw it was 300,000 wake-ups a second
      // (more CPU than the work); spinning 4,096 times before each sleep
      // instead was an eighth of its time. It spins briefly, and a sleeping
      // helper is woken for a batch of jobs (WakeBatch).
      bool arrived = false;
      for (int spin = 0; spin < 256 && !arrived; ++spin) {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ __volatile__("yield");
#endif
        arrived = h->produced.load(std::memory_order_acquire) != tail;
      }
      if (arrived)
        continue;
      if (h->stop.load(std::memory_order_acquire))
        return;
      std::unique_lock lock{h->mutex};
      h->sleeping.store(true, std::memory_order_seq_cst);
      h->idle.notify_all();
      h->work.wait(lock, [h] {
        return h->stop.load(std::memory_order_acquire) ||
               h->produced.load(std::memory_order_seq_cst) != h->consumed.load(std::memory_order_relaxed);
      });
      h->sleeping.store(false, std::memory_order_relaxed);
      continue;
    }
    const InterpJob& job = h->ring[tail % InterpHelper::Capacity];
    InterpRanges ranges{};
    const int steps = frame_interp::frame_steps();
    if (job.constantsCopied)
      std::memcpy(&h->constants, &job.constants, sizeof(h->constants));
    if (job.pixelCopied)
      std::memcpy(&h->pixel, &job.pixel, sizeof(h->pixel));
    if (frame_interp::blend_draw(job.input, h->constants, job.repeatsLastDraw, job.tev ? &h->pixel : nullptr) !=
        nullptr) {
      const bool repeated = frame_interp::last_blend_repeated();
      for (int step = 0; step < steps; ++step)
        ranges.uniform[step] =
            push_interp_uniform_dedup(h->cache[step], job.frameId, job.slot,
                                      reinterpret_cast<const uint8_t*>(frame_interp::blended_step(step)),
                                      sizeof(gxc::VertexShaderConstants), repeated);
    }
    for (int step = 0; step < steps; ++step) {
      const float* positions = frame_interp::blended_positions(step);
      const float* texcoords = frame_interp::blended_texcoords(step);
      if (positions != nullptr || texcoords != nullptr)
        ranges.verts[step] = push_blended_vertices(job.slot, job.vertices, positions, texcoords,
                                                 frame_interp::blended_texcoord_mask(), h->vertices);
      if (const gxc::PixelShaderConstants* pixel = frame_interp::blended_pixel(step))
        ranges.pixel[step] = push_interp_uniform_dedup(h->pixelCache[step], job.frameId, job.slot,
                                                       reinterpret_cast<const uint8_t*>(pixel), sizeof(*pixel), false);
    }
    resolve_interp_job(job.slot, ranges);
    h->consumed.store(tail + 1, std::memory_order_release);
  }
}

// Whether the game wrote this indexed draw's position array in this game
// frame or the one before. Per array (its guest address), a hash of the bytes
// its draws read, taken once a game frame, and the frame it last changed in;
// an array seen for the first time counts as written. The frame before counts
// too, so a mesh whose next positions are written before its draw is
// recorded is still taken as moving. (The guest's cache flushes, which give
// the texture cache its dirty epochs, miss small meshes the game writes
// without one: a flag on the Forsaken Fortress.)
// DOL_AURORA_INTERP_ALL_VERTICES=1 takes every such mesh as written.
bool positions_written(const gxc::DrawPlan& plan) {
  static const bool s_all = [] {
    const char* env = std::getenv("DOL_AURORA_INTERP_ALL_VERTICES");
    return env != nullptr && env[0] == '1';
  }();
  if (s_all || plan.match_direct_position || plan.match_position_span == 0 || plan.match_position_data == nullptr)
    return true;
  struct ArrayState {
    uint64_t hash = 0;
    uint64_t changed = 0; // the game frame its bytes last changed in
    uint64_t hashed = 0;  // the game frame they were last hashed in
    uint32_t span = 0;
  };
  static absl::flat_hash_map<uint32_t, ArrayState> s_arrays;
  const uint64_t frame = frame_interp::game_frame_number();
  // A model's draws mostly index one array in turn: the last answer, while
  // its frame holds.
  static uint32_t s_lastBase = 0, s_lastSpan = 0;
  static uint64_t s_lastFrame = UINT64_MAX;
  static bool s_lastWritten = true;
  if (plan.match_position_base == s_lastBase && plan.match_position_span <= s_lastSpan && frame == s_lastFrame)
    return s_lastWritten;
  auto [it, fresh] = s_arrays.try_emplace(plan.match_position_base);
  ArrayState& state = it->second;
  if (fresh || state.hashed != frame || plan.match_position_span > state.span) {
    // The widest span the array's draws have read (draws of one array reach
    // different lengths, in no fixed order), as far as this draw can read.
    const uint32_t span = std::min(std::max(plan.match_position_span, state.span),
                                   std::max(plan.match_position_readable, plan.match_position_span));
    const uint64_t hash = XXH3_64bits(plan.match_position_data, span);
    if (fresh || hash != state.hash || span != state.span)
      state.changed = frame;
    state.hash = hash;
    state.span = span;
    state.hashed = frame;
  }
  const bool written = state.changed + 1u >= frame;
  s_lastBase = plan.match_position_base;
  s_lastSpan = state.span;
  s_lastFrame = frame;
  s_lastWritten = written;
  if (s_arrays.size() > 65536u)
    s_arrays.clear(); // bounded; a cleared array counts as written once
  // DOL_AURORA_INTERP_VERTEX_LOG=1: how many arrays, and of them written, a
  // second (each array once per game frame).
  static const bool s_log = [] {
    const char* env = std::getenv("DOL_AURORA_INTERP_VERTEX_LOG");
    return env != nullptr && env[0] == '1';
  }();
  if (s_log) {
    static uint64_t s_logFrame = 0, s_seen = 0, s_written = 0;
    ++s_seen;
    s_written += written ? 1u : 0u;
    if (frame >= s_logFrame + 30u) {
      std::fprintf(stderr, "[interp-vertices] frame=%llu arrays=%llu written=%llu tracked=%zu\n",
                   static_cast<unsigned long long>(frame), static_cast<unsigned long long>(s_seen),
                   static_cast<unsigned long long>(s_written), s_arrays.size());
      s_logFrame = frame;
      s_seen = s_written = 0;
    }
  }
  return written;
}

uint32_t queue_interp_job(const gxc::DrawPlan& plan, const gxc::PixelShaderConstants& pixel, bool repeatsLastDraw,
                          bool tev, bool pixelRepeats, bool positionsWritten) {
  InterpHelper* h = g_interpHelper.load(std::memory_order_acquire);
  if (h == nullptr) {
    h = new InterpHelper;
    g_interpHelper.store(h, std::memory_order_release);
    std::thread(interp_helper_main, h).detach();
  }
  const uint64_t frameId = current_frame_id();
  if (h->packet != frameId) {
    h->packet = frameId;
    h->jobs = 0;
    h->pixelCarried = false;
  }
  const uint64_t head = h->produced.load(std::memory_order_relaxed);
  while (head - h->consumed.load(std::memory_order_acquire) >= InterpHelper::Capacity)
    std::this_thread::yield();
  InterpJob& job = h->ring[head % InterpHelper::Capacity];
  frame_interp::capture_draw(plan, job.input, positionsWritten);
  if (job.input.positions.empty())
    job.vertices.clear();
  else
    job.vertices.assign(plan.vertices.begin(), plan.vertices.end());
  job.frameId = frameId;
  job.slot = recording_frame_slot();
  job.repeatsLastDraw = repeatsLastDraw;
  // repeatsLastDraw: the constants are those last pushed in this frame packet,
  // and every push while frames are interpolated here queues a job, so they
  // are those of this packet's job before (the helper's copy). The first job
  // of a packet carries its own.
  job.constantsCopied = !repeatsLastDraw || h->jobs == 0u;
  frame_interp::note_job(job.constantsCopied);
  if (job.constantsCopied)
    std::memcpy(&job.constants, &plan.constants, sizeof(plan.constants));
  // pixelRepeats: the pixel constants are those last pushed in this frame
  // packet, which the last TEV job before this one carried (or repeated).
  job.tev = tev;
  job.pixelCopied = tev && (!pixelRepeats || !h->pixelCarried);
  if (job.pixelCopied) {
    std::memcpy(&job.pixel, &pixel, sizeof(pixel));
    h->pixelCarried = true;
  }
  h->produced.store(head + 1, std::memory_order_release);
  // Asleep, it consumes nothing, so a sleeping helper has WakeBatch or more
  // waiting before it is woken. Only then is `sleeping` read, after a full
  // fence: the helper sets it before its last look at `produced`, so one of
  // the two sees the other. Every job paid for a sequentially consistent store
  // and load here, which waited for the job's copies to leave the store buffer
  // (4 percent of the translation worker).
  if (head + 1 - h->consumed.load(std::memory_order_acquire) >= InterpHelper::WakeBatch) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (h->sleeping.load(std::memory_order_relaxed)) {
      std::lock_guard lock{h->mutex};
      h->work.notify_one();
    }
  }
  return h->jobs++;
}

// The job queue_interp_job gives the recording frame's next draw.
uint32_t next_interp_job() {
  const InterpHelper* h = g_interpHelper.load(std::memory_order_acquire);
  return h == nullptr || h->packet != current_frame_id() ? 0u : h->jobs;
}
} // namespace

void stop_interp_helper() {
  if (InterpHelper* h = g_interpHelper.load(std::memory_order_acquire)) {
    h->stop.store(true, std::memory_order_release);
    h->work.notify_all();
    g_interpHelper.store(nullptr, std::memory_order_release);
  }
}

void wait_interp_jobs() {
  InterpHelper* h = g_interpHelper.load(std::memory_order_acquire);
  if (h == nullptr ||
      h->consumed.load(std::memory_order_acquire) == h->produced.load(std::memory_order_acquire))
    return;
  std::unique_lock lock{h->mutex};
  // The last jobs, fewer than a batch, are waiting on this.
  h->work.notify_one();
  h->idle.wait(lock, [h] {
    return h->consumed.load(std::memory_order_acquire) == h->produced.load(std::memory_order_acquire);
  });
}

// known: 1 or 0 when the caller has already compared `data` with the cache's
// block (repeats_cached), -1 to compare here.
static Range push_uniform_dedup(UniformCache& cache, const uint8_t* data,
                                size_t length, int known = -1) {
  static const bool enabled = [] {
    const char* env = std::getenv("DOL_AURORA_UNIFORM_DEDUP");
    return env == nullptr || env[0] != '0';
  }();
  const uint64_t frameId = current_frame_id();
  if (enabled && frameId != 0 && cache.frameId == frameId && cache.range.size == length &&
      (known >= 0 ? known == 1
                  : std::memcmp(cached_bytes(cache, AURORA_UNIFORM_COMPARE_STAGED ? uniform_bytes(cache.range) : nullptr),
                                data, length) == 0)) {
    ++cache.hits;
    return cache.range;
  }
  ++cache.pushes;
  if (((cache.hits + cache.pushes) & 0x3FFFFu) == 0)
    Log.info("GXCore uniform reuse: {} of {} blocks reused",
             cache.hits, cache.hits + cache.pushes);
  cache.range = push_uniform(data, length);
  cache.frameId = frameId;
  keep_cached(cache, data, length);
  return cache.range;
}

// DOL_GXCORE_DRAW_DUMP=<game frame>: every draw of that frame, its TEV and
// indirect stages, texgens, textures and blending, on stderr (debug).
static void dump_draw(const gxc::DrawPlan& plan) {
  static const uint64_t frame = [] {
    const char* env = std::getenv("DOL_GXCORE_DRAW_DUMP");
    return env != nullptr ? std::strtoull(env, nullptr, 10) : 0ull;
  }();
  if (frame == 0 || frame_interp::game_frame_number() != frame)
    return;
  static uint32_t index = 0;
  const auto& key = plan.pipeline;
  const auto& sh = key.shader;
  const auto& m = plan.constants.posnormalmatrix;
  std::fprintf(stderr,
               "[draw-dump] #%u verts=%u prim=0x%02X t=(%.0f,%.0f,%.0f) blend=%u src=%u dst=%u z=%u/%u/%u tev=%u "
               "stages=%u ind=%u texgens=%u chans=%u lit=%u alpha=%u/%u/%u fog=%u tag=%06X\n",
               index++, plan.vertex_count, plan.match_primitive, m[0][3], m[1][3], m[2][3], key.blend_enable,
               key.src_factor, key.dst_factor, key.depth_test, key.depth_func, key.depth_update, sh.tev_valid,
               sh.num_tev_stages, sh.num_ind_stages, sh.num_tex_gens, sh.num_color_chans, sh.lit_valid,
               sh.alpha_comp0, sh.alpha_comp1, sh.alpha_logic, sh.fog_fsel, plan.draw_tag);
  for (uint32_t i = 0; i < sh.num_tev_stages && i < gxc::kMaxTevStages; ++i) {
    const auto& t = sh.tev_stages[i];
    std::fprintf(stderr,
                 "[draw-dump]   s%u tc%u map%u en%u ras%u  C=%u,%u,%u,%u op%u b%u s%u cl%u ->%u  "
                 "A=%u,%u,%u,%u op%u b%u s%u cl%u ->%u  k=%u/%u  ind: st%u fmt%u bias%u mtx%u id%u wrap%u,%u add%u lod%u\n",
                 i, t.tevorders_texcoord, t.tevorders_texmap, t.tevorders_enable, t.tevorders_colorchan, t.cc_a,
                 t.cc_b, t.cc_c, t.cc_d, t.cc_op, t.cc_bias, t.cc_scale, t.cc_clamp, t.cc_dest, t.ac_a, t.ac_b, t.ac_c,
                 t.ac_d, t.ac_op, t.ac_bias, t.ac_scale, t.ac_clamp, t.ac_dest, t.ksel_kc, t.ksel_ka, t.ind_stage,
                 t.ind_format, t.ind_bias, t.ind_matrix_index, t.ind_matrix_id, t.ind_wrap_s, t.ind_wrap_t,
                 t.ind_add_prev, t.ind_use_original_lod);
  }
  for (uint32_t i = 0; i < sh.num_ind_stages && i < 4u; ++i)
    std::fprintf(stderr, "[draw-dump]   ind%u map%u tc%u scale=%u,%u\n", i, sh.ind_stages[i].texmap,
                 sh.ind_stages[i].texcoord, sh.ind_stages[i].scale_s, sh.ind_stages[i].scale_t);
  for (uint32_t i = 0; i < sh.num_tex_gens && i < gxc::kMaxTexGens; ++i) {
    std::fprintf(stderr, "[draw-dump]   tg%u type%u src%u form%u proj%u row%u (MatrixIndexA %08X)\n", i,
                 sh.tex_gens[i].texgentype, sh.tex_gens[i].sourcerow, sh.tex_gens[i].inputform,
                 sh.tex_gens[i].projection, plan.texgen_row[i], plan.matrix_index_a);
    for (uint32_t r = 0; r < 3u; ++r) {
      const float* row = plan.constants.texmatrices[i * 3u + r];
      std::fprintf(stderr, "[draw-dump]     tm%u.%u %.4f %.4f %.4f %.4f\n", i, r, row[0], row[1], row[2], row[3]);
    }
  }
  if (plan.vertices.size() >= 3)
    std::fprintf(stderr, "[draw-dump]   v0 %.2f %.2f %.2f\n", plan.vertices[0], plan.vertices[1], plan.vertices[2]);
  for (uint32_t r = 0; r < 3u; ++r)
    std::fprintf(stderr, "[draw-dump]   pos%u %.4f %.4f %.4f %.4f\n", r, m[r][0], m[r][1], m[r][2], m[r][3]);
  for (uint32_t i = 0; i < 8u; ++i)
    if (plan.textures[i].valid)
      std::fprintf(stderr, "[draw-dump]   tex%u @%08X fmt%u %ux%u dims=%d,%d,%d,%d\n", i, plan.textures[i].address,
                   plan.textures[i].format, plan.textures[i].width, plan.textures[i].height,
                   plan.pixel_constants.texdims[i][0], plan.pixel_constants.texdims[i][1],
                   plan.pixel_constants.texdims[i][2], plan.pixel_constants.texdims[i][3]);
  if (sh.num_ind_stages != 0)
    for (uint32_t i = 0; i < 6u; ++i)
      std::fprintf(stderr, "[draw-dump]   indmtx%u %d %d %d %d\n", i, plan.pixel_constants.indtexmtx[i][0],
                   plan.pixel_constants.indtexmtx[i][1], plan.pixel_constants.indtexmtx[i][2],
                   plan.pixel_constants.indtexmtx[i][3]);
  for (uint32_t i = 0; i < 4u; ++i)
    std::fprintf(stderr, "[draw-dump]   c%u=%d,%d,%d,%d k%u=%d,%d,%d,%d\n", i, plan.pixel_constants.colors[i][0],
                 plan.pixel_constants.colors[i][1], plan.pixel_constants.colors[i][2], plan.pixel_constants.colors[i][3],
                 i, plan.pixel_constants.kcolors[i][0], plan.pixel_constants.kcolors[i][1],
                 plan.pixel_constants.kcolors[i][2], plan.pixel_constants.kcolors[i][3]);
}

// The sun shadows' facing term (post_fx::sun_normals_wanted()) shades by a smooth normal where the scenery
// has one, as Wind Waker HD's models light per vertex. The GameCube's scenery has none (it's lit by baked
// vertex colours), so a solid scenery draw without normals gets them here: each vertex the sum of the
// triangles' normals (by their area) at its position, as the model repeats a vertex for each face (its
// texture coordinate or colour differs), short of a crease (smooth_crease_cos()). Not a character's, the
// sea's, or a skinned draw's. Off, nothing is done.
static bool smooth_normal_draw(const gxc::DrawPlan& plan, bool sea) {
  const auto& shader = plan.pipeline.shader;
  // Skinned scenery (a palm) too, its cut-out leaves included: its positions are each bone's own, and the mask's
  // pass turns the normals by the same matrices.
  const bool skinned = shader.has_pos_mtx_idx != 0u && skinned_scenery() && !toon_draw(plan.pipeline);
  const bool solid = shader.has_pos_mtx_idx == 0u && sun_caster(plan.pipeline) && !character_draw(plan.pipeline);
  return post_fx::sun_normals_wanted() && !sea && plan.constants.projection[3][3] != 1.f &&
         plan.pipeline.depth_update != 0u && shader.has_vertex_normal == 0u && shader.raw_tex_hi_in_nbt == 0u &&
         (solid || (skinned && (sun_caster(plan.pipeline) || sun_caster_alpha(plan.pipeline)))) &&
         plan.indices.size() >= 3 && plan.vertices.size() >= gxc::kVertexFloats * 3u;
}

// A crease: faces meeting at a vertex more than this many degrees apart keep their own normals there (a
// wall and the flat top above it), so a sunlit top doesn't light the wall below it
// (DOL_AURORA_SUN_SHADOW_FACING_CREASE, 60 by default).
static float smooth_crease_cos() {
  static const float value = [] {
    const char* env = std::getenv("DOL_AURORA_SUN_SHADOW_FACING_CREASE");
    const float degrees = env != nullptr ? static_cast<float>(std::atof(env)) : 60.f;
    return std::cos(std::clamp(degrees, 0.f, 180.f) * 3.14159265f / 180.f);
  }();
  return value;
}

// Wind Waker HD's own normals for the scenery, by the GameCube's exact vertex position: a stage's table from the
// HD pack, given by the host as the stage changes (aurora_set_hd_normals), or one fixed table for tests
// (DOL_AURORA_SUN_SHADOW_FACING_HD_NORMALS). Format: b'BWHN', u32 version 1, u32 count, then count x f32 x, y,
// z, nx, ny, nz; a position at one of HD's hard edges is listed once per normal. HD remade its models with the
// GameCube's positions and normals of its own, which its per-vertex light follows.
struct HdNormals {
  absl::flat_hash_map<std::array<uint32_t, 3>, std::pair<uint32_t, uint32_t>> at; // first, count in `normals`
  std::vector<std::array<float, 3>> normals;
};

static std::shared_ptr<const HdNormals> load_hd_normals(const char* path) {
  {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr)
      return nullptr;
    char magic[4]{};
    uint32_t version = 0, count = 0;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "BWHN", 4) == 0 &&
              std::fread(&version, 4, 1, f) == 1 && version == 1 && std::fread(&count, 4, 1, f) == 1;
    std::vector<std::array<float, 6>> rows(ok ? count : 0);
    ok = ok && std::fread(rows.data(), sizeof(rows[0]), rows.size(), f) == rows.size();
    std::fclose(f);
    if (!ok)
      return nullptr;
    auto table = std::make_shared<HdNormals>();
    table->normals.reserve(rows.size());
    for (const auto& r : rows) {
      std::array<uint32_t, 3> key{};
      std::memcpy(key.data(), r.data(), sizeof(key));
      auto [it, added] = table->at.try_emplace(key, static_cast<uint32_t>(table->normals.size()), 0u);
      if (!added && it->second.first + it->second.second != table->normals.size())
        continue; // a position listed apart: keep its first run
      table->normals.push_back({r[3], r[4], r[5]});
      ++it->second.second;
    }
    std::fprintf(stderr, "[sun-facing] HD normals: %zu positions from %s\n", table->at.size(), path);
    return table;
  }
}

std::mutex g_hdNormalsMutex;
std::shared_ptr<const HdNormals> g_hdNormals;  // the stage's (aurora_set_hd_normals)

// The table in use: the test's fixed one, else the stage's.
static std::shared_ptr<const HdNormals> hd_normals() {
  static const std::shared_ptr<const HdNormals> fixed = []() -> std::shared_ptr<const HdNormals> {
    const char* path = std::getenv("DOL_AURORA_SUN_SHADOW_FACING_HD_NORMALS");
    return path != nullptr && path[0] != '\0' ? load_hd_normals(path) : nullptr;
  }();
  if (fixed)
    return fixed;
  std::lock_guard lock{g_hdNormalsMutex};
  return g_hdNormals;
}

// Wind Waker HD's normals for the game's object models (the HD pack's optional/hd-normals/objects.bin, from the
// importer; aurora_set_hd_object_tables), the same format, looked up after the stage's.
std::shared_ptr<const HdNormals> g_hdObjectNormals;

static std::shared_ptr<const HdNormals> hd_object_normals() {
  std::lock_guard lock{g_hdNormalsMutex};
  return g_hdObjectNormals;
}

// Wind Waker HD's own vertex colours, by the same exact positions: the objects' (the HD pack's
// optional/hd-colours/objects.bin; aurora_set_hd_object_tables), or one fixed table for tests
// (DOL_AURORA_HD_VERTEX_COLOURS). Format: b'BWHC', u32 version 1, u32 count, then count x f32 x, y, z, r, g, b, a.
// A stage's table with Wind Waker HD's lava glow is version 2: u32 count, u32 curves, each curve's u32 frame count
// and its frames' f32 linear r, g, b (HD's konst2 animation, the lava's glow colour, at each game frame), then count
// x f32 x, y, z, r, g, b, a, glow (HD's vertex red squared), curve (-1 none).
// HD lights its palms as texture x vertex colour x (sky-to-ground ambient + sun x N.L x shadow); the sun pass makes
// the light, so the draw gives texture x vertex colour x the light colour: the GameCube's palms go from the shade
// colour (C0) to the light one (KONST) by their vertex colour, so such a draw takes C0 black and HD's colour (a
// vertex the table hasn't keeps its own result). Only in the draws that take smooth normals, so only while the HD
// sun shadows' facing term is on.
// A row: r, g, b, a, glow, its curve (-1 none).
struct HdColours : absl::flat_hash_map<std::array<uint32_t, 3>, std::array<float, 6>> {
  std::vector<std::vector<std::array<float, 3>>> curves; // the lava's glow colour at each game frame, linear
};

// gamma: the stage's rooms' colours (aurora_set_hd_stage_colours), which HD multiplies raw in linear light, taken to
// the game's gamma (^ 1/2.2) for the TEV, which multiplies in it.
static std::shared_ptr<const HdColours> load_hd_colours(const char* path, bool gamma = false) {
  FILE* f = std::fopen(path, "rb");
  if (f == nullptr)
    return nullptr;
  char magic[4]{};
  uint32_t version = 0, count = 0, curveCount = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "BWHC", 4) == 0 &&
            std::fread(&version, 4, 1, f) == 1 && (version == 1 || version == 2) &&
            std::fread(&count, 4, 1, f) == 1 && count < (1u << 24);
  auto out = std::make_shared<HdColours>();
  if (ok && version == 2) {
    ok = std::fread(&curveCount, 4, 1, f) == 1 && curveCount < 256;
    for (uint32_t c = 0; ok && c < curveCount; ++c) {
      uint32_t frames = 0;
      ok = std::fread(&frames, 4, 1, f) == 1 && frames > 0 && frames <= 4096;
      std::vector<std::array<float, 3>> curve(ok ? frames : 0);
      ok = ok && std::fread(curve.data(), sizeof(curve[0]), curve.size(), f) == curve.size();
      out->curves.push_back(std::move(curve));
    }
  }
  const size_t floats = version == 2 ? 9 : 7;
  std::vector<float> rows(ok ? size_t(count) * floats : 0);
  ok = ok && std::fread(rows.data(), sizeof(float), rows.size(), f) == rows.size();
  std::fclose(f);
  if (!ok)
    return nullptr;
  for (size_t i = 0; i < count; ++i) {
    const float* r = &rows[i * floats];
    std::array<uint32_t, 3> key{};
    std::memcpy(key.data(), r, sizeof(key));
    const auto g = [&](float v) { return gamma ? std::pow(std::clamp(v, 0.f, 1.f), 1.f / 2.2f) : v; };
    const float curve = floats == 9 && r[8] >= 0.f && r[8] < static_cast<float>(out->curves.size()) ? r[8] : -1.f;
    out->try_emplace(key, std::array<float, 6>{g(r[3]), g(r[4]), g(r[5]), r[6], curve >= 0.f ? r[7] : 0.f, curve});
  }
  std::fprintf(stderr, "[sun-facing] HD vertex colours: %zu positions from %s\n", out->size(), path);
  return out;
}

std::shared_ptr<const HdColours> g_hdObjectColours;  // aurora_set_hd_object_tables
std::shared_ptr<const HdColours> g_hdStageColours;   // aurora_set_hd_stage_colours, in the game's gamma

// The stage's rooms' colour for a position, then the objects'.
static const std::array<float, 6>* hd_colour(const HdColours* stage, const HdColours* objects,
                                             const std::array<uint32_t, 3>& key) {
  if (stage != nullptr)
    if (const auto it = stage->find(key); it != stage->end())
      return &it->second;
  if (objects != nullptr)
    if (const auto it = objects->find(key); it != objects->end())
      return &it->second;
  return nullptr;
}

static std::shared_ptr<const HdColours> hd_stage_colours() {
  std::lock_guard lock{g_hdNormalsMutex};
  return g_hdStageColours;
}

static std::shared_ptr<const HdColours> hd_colours() {
  static const std::shared_ptr<const HdColours> fixed = []() -> std::shared_ptr<const HdColours> {
    const char* path = std::getenv("DOL_AURORA_HD_VERTEX_COLOURS");
    return path != nullptr && path[0] != '\0' ? load_hd_colours(path) : nullptr;
  }();
  if (fixed)
    return fixed;
  std::lock_guard lock{g_hdNormalsMutex};
  return g_hdObjectColours;
}

// A konst colour selector's rgb (0..255).
static std::array<int, 3> konst_rgb(const gxc::PixelShaderConstants& pc, uint8_t sel) {
  if (sel < 8) {
    const int v = sel == 0 ? 255 : 255 - sel * 32 + (sel >= 4 ? 1 : 0);
    return {v, v, v};
  }
  if (sel < 12)
    return {0, 0, 0};
  if (sel < 16)
    return {pc.kcolors[sel - 12][0], pc.kcolors[sel - 12][1], pc.kcolors[sel - 12][2]};
  const int v = pc.kcolors[sel & 3][((sel - 16) >> 2) & 3];
  return {v, v, v};
}

// Whether a draw takes HD's light: its first stage is lerp(C0, KONST, RASC) from colour channel 0 (the palms'), no
// later stage reads C0's colour, and the table has some of its positions.
// DOL_AURORA_SUN_SHADOW_HD_TERRAIN_LIGHT=1 (local test): every such draw takes HD's light, the scenery's too, its own
// vertex colour kept where no table has HD's. HD's terrain material is the palms' rule, albedo x vertex colour x
// mix(ambient, sun, N.L x shadow), and its vertex colours are the GameCube's baked greys (Cemu probe v4, Oct 30): the
// GameCube's mix never went below the shade colour, so its greys under 1 stood lighter than HD's.
static bool hd_terrain_light() {
  static const bool on = [] {
    const char* env = std::getenv("DOL_AURORA_SUN_SHADOW_HD_TERRAIN_LIGHT");
    return env != nullptr && env[0] == '1';
  }();
  return on;
}

static bool hd_light_draw(const gxc::DrawPlan& plan) {
  const std::shared_ptr<const HdColours> table = hd_colours();
  const auto& shader = plan.pipeline.shader;
  if ((table == nullptr && !hd_terrain_light()) || shader.tev_valid == 0u || shader.num_tev_stages == 0u)
    return false;
  const gxc::TevStageKey& s = shader.tev_stages[0];
  // a C0, b KONST, c RASC, d ZERO; add, no bias or scale; colour channel 0
  if (s.cc_a != 2 || s.cc_b != 14 || s.cc_c != 10 || s.cc_d != 15 || s.cc_bias != 0 || s.cc_op != 0 ||
      s.cc_scale != 0 || (s.tevorders_colorchan != 0 && s.tevorders_colorchan != 4))
    return false;
  for (uint32_t n = 1; n < shader.num_tev_stages; ++n) {
    const gxc::TevStageKey& t = shader.tev_stages[n];
    for (const uint8_t arg : {t.cc_a, t.cc_b, t.cc_c, t.cc_d})
      if (arg == 2 || arg == 3)
        return false;
  }
  if (hd_terrain_light())
    return true;
  const size_t count = plan.vertices.size() / gxc::kVertexFloats;
  for (size_t v = 0; v < count; ++v) {
    std::array<uint32_t, 3> key{};
    std::memcpy(key.data(), &plan.vertices[v * gxc::kVertexFloats + gxc::kVertexPosOffset / sizeof(float)],
                sizeof(key));
    if (table->contains(key))
      return true;
  }
  return false;
}

// Such a draw's pixel constants: C0's colour black; C1's too for a lava draw with the GameCube's own glow
// (hd_lava_gc_glow()), which HD's shaders for those materials haven't.
static const gxc::PixelShaderConstants& hd_light_constants(const gxc::DrawPlan& plan, bool noGcGlow) {
  static thread_local gxc::PixelShaderConstants pc;
  pc = plan.pixel_constants;
  pc.colors[1][0] = pc.colors[1][1] = pc.colors[1][2] = 0;
  if (noGcGlow)
    pc.colors[2][0] = pc.colors[2][1] = pc.colors[2][2] = 0;
  return pc;
}

// A draw's pixel constants with Wind Waker HD's fog start before the camera (gxc::fog_start_before_camera()), while
// the host has HD's fog in the game's palettes and a start below 0 for the frame; else `pc` itself.
static const gxc::PixelShaderConstants& hd_fog_constants(const gxc::PixelShaderConstants& pc,
                                                         const gxc::DrawPlan& plan) {
  const gxc::ShaderKey& key = plan.pipeline.shader;
  if (key.fog_fsel != static_cast<uint8_t>(gxc::FogType::Linear) || pc.fogf[1] != 0.f || !post_fx::fog_hd())
    return pc;
  float start = 0.f, end = 0.f;
  if (!post_fx::fog_hd_start(start, end))
    return pc;
  static thread_local gxc::PixelShaderConstants out;
  out = pc;
  return gxc::fog_start_before_camera(out, key, plan.constants.projection, start, end) ? out : pc;
}

// Wind Waker HD's lava and fire rooms (GanonB, Cave01, MiniKaz, Ganon's Tower's lava parts): their materials with
// a konst2 animation add the lava's light, albedo x lava K x 4 x the vertex red squared (Cemu, GanonB's PS 4345;
// lava K the animation at the game frame, 0.453, 0.0415, 0.0005 at frame 23), outside the shade and before the fog.
// The stage's table (version 2) has each such vertex's glow and curve. A lit room draw in the GameCube's two-stage
// form (the light, then x the texture: every such material's) with HD's light buffer on takes it: its vertices'
// colour 1 (no stage reads it) carry the glow's light, linear, and in alpha -(1 + its share of the lit light), which
// the sea mask's pass puts in the normals' alpha so the sun pass leaves that share unshaded.
// Dragon Roost Cavern's (M_NewD2 rooms 2, 7, 8, 10-13, 15; the disc's model.sharcfb) add lava K x 4 x (1 - the
// vertex alpha) squared, the importer's glow there, by the same rule. Its _v_x materials are three stages on the
// GameCube, a third PREV + C1 x APREV (the GameCube's own glow, hd_lava_gc_glow()); HD's shaders for them are their
// plain materials' (the same program, no such term), so the draw takes C1 black (hd_light_constants()) and the glow
// goes on after the texture's stage (gxcore_shader.cpp emit_hd_lava()).
static bool hd_lava_gc_glow(const gxc::ShaderKey& s) {
  if (s.num_tev_stages != 3u)
    return false;
  const gxc::TevStageKey& t = s.tev_stages[2];
  // a ZERO, b C1, c APREV, d CPREV; add, no bias or scale, into PREV
  return t.cc_a == 15u && t.cc_b == 4u && t.cc_c == 1u && t.cc_d == 0u && t.cc_bias == 0u && t.cc_op == 0u &&
         t.cc_scale == 0u && t.cc_dest == 0u;
}

static bool hd_lava_shape(const gxc::ShaderKey& s) {
  if ((s.num_tev_stages != 2u && !hd_lava_gc_glow(s)) || s.num_color_chans > 1u)
    return false;
  const gxc::TevStageKey& t = s.tev_stages[1];
  return t.cc_a == 15u && t.cc_d == 15u && ((t.cc_b == 0u && t.cc_c == 8u) || (t.cc_b == 8u && t.cc_c == 0u)) &&
         t.cc_bias == 0u && t.cc_op == 0u && t.cc_scale == 0u && t.cc_dest == 0u;
}

static bool hd_lava_draw(const gxc::DrawPlan& plan) {
  if (!hd_terrain_light() || !hd_lava_shape(plan.pipeline.shader) || !hd_light_draw(plan))
    return false;
  const std::shared_ptr<const HdColours> stage = hd_stage_colours();
  if (stage == nullptr || stage->curves.empty())
    return false;
  const size_t count = plan.vertices.size() / gxc::kVertexFloats;
  for (size_t v = 0; v < count; ++v) {
    std::array<uint32_t, 3> key{};
    std::memcpy(key.data(), &plan.vertices[v * gxc::kVertexFloats + gxc::kVertexPosOffset / sizeof(float)],
                sizeof(key));
    if (const auto it = stage->find(key); it != stage->end() && it->second[5] >= 0.f)
      return true;
  }
  return false;
}

// The draw being uploaded takes the lava's glow (submit_draw_plan(), read by apply_hd_colours()).
static thread_local bool t_hdLavaDraw = false;

// Such a draw's vertex colours: HD's, and, where the table hasn't the position, the fraction of the light colour
// the GameCube's mix gave (shade + (light - shade) x own) / light; with HD's terrain light, its own.
static void apply_hd_colours(const gxc::DrawPlan& plan, std::vector<float>& vertices) {
  if (!hd_light_draw(plan))
    return;
  const std::shared_ptr<const HdColours> table = hd_colours();
  // With HD's terrain light, the stage's rooms' own colours first (HD remade them).
  const std::shared_ptr<const HdColours> stage = hd_terrain_light() ? hd_stage_colours() : nullptr;
  if (table == nullptr && stage == nullptr)
    return; // HD's terrain light without tables: the draw's own vertex colours
  const gxc::TevStageKey& s = plan.pipeline.shader.tev_stages[0];
  const auto& pc = plan.pixel_constants;
  const std::array<int, 3> k = konst_rgb(pc, s.ksel_kc);
  const std::array<int, 3> c{pc.colors[1][0], pc.colors[1][1], pc.colors[1][2]};
  static const bool log = std::getenv("DOL_AURORA_SUN_SHADOW_FACING_LOG") != nullptr;
  static thread_local uint64_t s_draws = 0;
  const size_t count = vertices.size() / gxc::kVertexFloats;
  size_t hits = 0;
  const uint64_t frame = frame_interp::game_frame_number();
  for (size_t v = 0; v < count; ++v) {
    float* vert = &vertices[v * gxc::kVertexFloats];
    std::array<uint32_t, 3> key{};
    std::memcpy(key.data(), vert + gxc::kVertexPosOffset / sizeof(float), sizeof(key));
    float* col = vert + gxc::kVertexColor0Offset / sizeof(float);
    const std::array<float, 6>* hd = hd_colour(stage.get(), table.get(), key);
    if (hd != nullptr)
      ++hits;
    for (int ch = 0; ch < 3; ++ch) {
      if (hd != nullptr) {
        col[ch] = (*hd)[ch];
      } else if (k[ch] > 0 && !hd_terrain_light()) {
        col[ch] = std::clamp((c[ch] + (k[ch] - c[ch]) * col[ch]) / static_cast<float>(k[ch]), 0.f, 1.f);
      }
    }
    if (t_hdLavaDraw) {
      // The lava's glow (hd_lava_draw()): lava K x 4 x the glow, and its share of the lit light, K x the colour
      // (the sun pass's lit light, the lamps left out), by luminance.
      float* col1 = vert + gxc::kVertexColor1Offset / sizeof(float);
      std::array<float, 3> glow{0.f, 0.f, 0.f};
      if (hd != nullptr && (*hd)[5] >= 0.f && stage != nullptr) {
        const auto& curve = stage->curves[static_cast<size_t>((*hd)[5])];
        const auto& lava = curve[static_cast<size_t>(frame % curve.size())];
        for (int ch = 0; ch < 3; ++ch)
          glow[ch] = 4.f * lava[ch] * (*hd)[4];
      }
      constexpr std::array<float, 3> luma{0.2126f, 0.7152f, 0.0722f};
      float lit = 0.f, glowLuma = 0.f;
      for (int ch = 0; ch < 3; ++ch) {
        lit += luma[ch] * std::pow(std::clamp(static_cast<float>(k[ch]) / 255.f * col[ch], 0.f, 1.f), 2.2f);
        glowLuma += luma[ch] * glow[ch];
      }
      const float share = glowLuma > 0.f ? glowLuma / (lit + glowLuma) : 0.f;
      col1[0] = glow[0];
      col1[1] = glow[1];
      col1[2] = glow[2];
      col1[3] = -1.f - share;
    }
  }
  if (log && ++s_draws % 600 == 1)
    std::fprintf(stderr, "[sun-facing] HD light: %zu of %zu vertices HD's colours, shade %d,%d,%d light %d,%d,%d\n",
                 hits, count, c[0], c[1], c[2], k[0], k[1], k[2]);
}

static const std::vector<float>& smooth_normals_uncached(const gxc::DrawPlan& plan) {
  using Vec = std::array<float, 3>;
  static thread_local std::vector<float> vertices;
  static thread_local std::vector<uint32_t> group;       // each vertex's position
  static thread_local std::vector<Vec> own;              // each vertex's own faces' normals, summed
  static thread_local std::vector<Vec> faceSum, faceUnit; // each face's normal (by its area) and direction
  static thread_local std::vector<uint32_t> groupStart, groupFaces, groupFill;
  static thread_local absl::flat_hash_map<std::array<uint32_t, 3>, uint32_t> weld;
  vertices = plan.vertices;
  const size_t count = vertices.size() / gxc::kVertexFloats;
  group.resize(count);
  own.assign(count, Vec{0.f, 0.f, 0.f});
  faceSum.clear();
  faceUnit.clear();
  weld.clear();
  for (size_t v = 0; v < count; ++v) {
    const float* p = &vertices[v * gxc::kVertexFloats + gxc::kVertexPosOffset / sizeof(float)];
    std::array<uint32_t, 3> key{};
    std::memcpy(key.data(), p, sizeof(key));
    group[v] = weld.try_emplace(key, static_cast<uint32_t>(weld.size())).first->second;
  }
  const size_t groups = weld.size();
  const auto pos = [&](uint16_t v) {
    return &vertices[size_t{v} * gxc::kVertexFloats + gxc::kVertexPosOffset / sizeof(float)];
  };
  const auto valid = [&](size_t i) {
    return plan.indices[i] < count && plan.indices[i + 1] < count && plan.indices[i + 2] < count;
  };
  groupStart.assign(groups + 1, 0u);
  for (size_t i = 0; i + 2 < plan.indices.size(); i += 3) {
    if (!valid(i)) {
      faceSum.push_back({0.f, 0.f, 0.f});
      faceUnit.push_back({0.f, 0.f, 0.f});
      continue;
    }
    const uint16_t a = plan.indices[i], b = plan.indices[i + 1], c = plan.indices[i + 2];
    const float* pa = pos(a);
    const float* pb = pos(b);
    const float* pc = pos(c);
    const float e1[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
    const float e2[3] = {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
    const Vec n{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const float length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    faceSum.push_back(n);
    faceUnit.push_back(length > 0.f ? Vec{n[0] / length, n[1] / length, n[2] / length} : Vec{0.f, 0.f, 0.f});
    for (const uint16_t v : {a, b, c}) {
      for (int k = 0; k < 3; ++k)
        own[v][k] += n[k];
      ++groupStart[group[v] + 1];
    }
  }
  // Each position's faces, in one list (groupStart[g] .. groupStart[g + 1]).
  for (size_t g = 0; g < groups; ++g)
    groupStart[g + 1] += groupStart[g];
  groupFaces.resize(groupStart[groups]);
  groupFill.assign(groupStart.begin(), groupStart.end() - 1);
  for (size_t i = 0, f = 0; i + 2 < plan.indices.size(); i += 3, ++f)
    if (valid(i))
      for (int corner = 0; corner < 3; ++corner)
        groupFaces[groupFill[group[plan.indices[i + corner]]]++] = static_cast<uint32_t>(f);
  // A vertex's normal: HD's at its position when the table has it (the one nearest its own faces' at one of
  // HD's hard edges), else the faces at its position that turn within the crease of its own faces.
  const float crease = smooth_crease_cos();
  // The stage's table first, then the objects' (the same position in a stage model and an object is the stage's).
  const std::shared_ptr<const HdNormals> hdTable = hd_normals();
  const std::shared_ptr<const HdNormals> objectTable = hd_object_normals();
  const std::array<const HdNormals*, 2> tables{hdTable.get(), objectTable.get()};
  // DOL_AURORA_SUN_SHADOW_FACING_LOG=1 (local debug): how many scenery vertices found HD's normal.
  static const bool log = std::getenv("DOL_AURORA_SUN_SHADOW_FACING_LOG") != nullptr;
  static thread_local uint64_t s_hdHits = 0, s_vertices = 0, s_calls = 0;
  s_vertices += count;
  if (log && (tables[0] != nullptr || tables[1] != nullptr) && ++s_calls % 4000 == 0)
    std::fprintf(stderr, "[sun-facing] HD normals for %llu of %llu scenery vertices\n",
                 static_cast<unsigned long long>(s_hdHits), static_cast<unsigned long long>(s_vertices));
  for (size_t v = 0; v < count; ++v) {
    float* out = &vertices[v * gxc::kVertexFloats + gxc::kVertexNormalOffset / sizeof(float)];
    const Vec& ref = own[v];
    const float refLength = std::sqrt(ref[0] * ref[0] + ref[1] * ref[1] + ref[2] * ref[2]);
    std::array<uint32_t, 3> key{};
    std::memcpy(key.data(), &vertices[v * gxc::kVertexFloats + gxc::kVertexPosOffset / sizeof(float)], sizeof(key));
    bool fromHd = false;
    for (const HdNormals* hd : tables) {
      if (hd == nullptr)
        continue;
      if (const auto it = hd->at.find(key); it != hd->at.end()) {
        const Vec* best = &hd->normals[it->second.first];
        float bestDot = -2.f;
        for (uint32_t j = 0; j < it->second.second; ++j) {
          const Vec& c = hd->normals[it->second.first + j];
          // The model's winding isn't known: the mask's pass turns the normal to the side seen.
          const float d = std::abs(c[0] * ref[0] + c[1] * ref[1] + c[2] * ref[2]);
          if (d > bestDot) {
            bestDot = d;
            best = &c;
          }
        }
        for (int k = 0; k < 3; ++k)
          out[k] = (*best)[k];
        ++s_hdHits;
        fromHd = true;
        break;
      }
    }
    if (fromHd)
      continue;
    Vec n{0.f, 0.f, 0.f};
    if (refLength > 0.f) {
      for (uint32_t j = groupStart[group[v]]; j < groupStart[group[v] + 1]; ++j) {
        const uint32_t f = groupFaces[j];
        const Vec& u = faceUnit[f];
        if ((u[0] * ref[0] + u[1] * ref[1] + u[2] * ref[2]) / refLength >= crease)
          for (int k = 0; k < 3; ++k)
            n[k] += faceSum[f][k];
      }
    }
    const float length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int k = 0; k < 3; ++k)
      out[k] = length > 0.f ? n[k] / length : 0.f;
  }
  apply_hd_colours(plan, vertices);
  return vertices;
}

// What smooth_normals_uncached() reads besides the draw's vertices and indices, for SmoothNormalCache's compare: the
// stage's and the objects' tables (a new table is a new pointer; the entries hold the old one), and whether the draw
// takes HD's light, with apply_hd_colours()'s shade and light colours if so.
struct SmoothNormalInputs {
  std::shared_ptr<const HdNormals> normals, objectNormals;
  std::shared_ptr<const HdColours> colours, stageColours;
  bool light = false;
  std::array<int, 3> k{}, c{};
  bool operator==(const SmoothNormalInputs& o) const {
    return normals == o.normals && objectNormals == o.objectNormals && colours == o.colours &&
           stageColours == o.stageColours && light == o.light && k == o.k && c == o.c;
  }
};

// The draw's vertices with smooth normals, made once while the draw stays the same (smooth_normal_cache.hpp): a
// still room's lit scenery made them every frame, Hyrule Castle's hall 66 draws and 47,000 vertices a game frame
// (on an i5-6500, 7 ms of the GX worker's 20). Not for a lava draw (its glow follows the game frame).
// DOL_AURORA_SUN_SHADOW_NORMAL_CACHE=0: made every frame, as before.
// DOL_AURORA_SUN_SHADOW_NORMAL_CACHE_LOG=1 (local debug): draws, vertices and hits a game frame, every 600 frames.
static const std::vector<float>& with_smooth_normals(const gxc::DrawPlan& plan) {
  static const bool cacheOn = [] {
    const char* env = std::getenv("DOL_AURORA_SUN_SHADOW_NORMAL_CACHE");
    return env == nullptr || env[0] != '0';
  }();
  if (!cacheOn || t_hdLavaDraw)
    return smooth_normals_uncached(plan);
  static thread_local SmoothNormalCache<SmoothNormalInputs> cache;
  const uint64_t frame = frame_interp::game_frame_number();
  SmoothNormalInputs inputs{hd_normals(), hd_object_normals(), hd_colours(),
                            hd_terrain_light() ? hd_stage_colours() : nullptr};
  // apply_hd_colours()'s own tests and colours.
  if ((inputs.colours != nullptr || inputs.stageColours != nullptr) && hd_light_draw(plan)) {
    inputs.light = true;
    const gxc::TevStageKey& s = plan.pipeline.shader.tev_stages[0];
    inputs.k = konst_rgb(plan.pixel_constants, s.ksel_kc);
    inputs.c = {plan.pixel_constants.colors[1][0], plan.pixel_constants.colors[1][1],
                plan.pixel_constants.colors[1][2]};
  }
  static const bool log = std::getenv("DOL_AURORA_SUN_SHADOW_NORMAL_CACHE_LOG") != nullptr;
  if (log) {
    static thread_local uint64_t s_vertices = 0, s_frames = 0, s_logFrame = 0;
    s_vertices += plan.vertices.size() / gxc::kVertexFloats;
    if (frame != s_logFrame) {
      s_logFrame = frame;
      if (++s_frames % 600 == 0) {
        const auto [draws, hits] = cache.take_counts();
        std::fprintf(stderr, "[sun-normals] a game frame: %.0f draws, %.0f vertices, %.1f%% cache hits, %zu cached\n",
                     draws / 600.0, s_vertices / 600.0, draws ? 100.0 * hits / draws : 0.0, cache.size());
        s_vertices = 0;
      }
    }
  }
  return cache.get(plan.vertices, plan.indices, std::move(inputs), frame,
                   [&]() -> const std::vector<float>& { return smooth_normals_uncached(plan); });
}

bool submit_draw_plan(const gxc::DrawPlan& plan) {
  if (!plan.ok || plan.vertex_count == 0 || plan.indices.empty()) {
    return false;
  }
  dump_draw(plan);

  if (plan.viewport_valid) {
    gx::set_logical_viewport(retail_viewport(plan.viewport));
  }
  if (plan.scissor_valid) {
    // Wind Waker draws its 2D layer (HUD, menus and the fade to and from
    // black) with a 640x479 scissor, so the frame's last line keeps the
    // previous scene through every fade. A television hides that line in its
    // overscan; an iPad shows it as a strip of the old area along the bottom
    // during transitions. Extend that full-width scissor to the full frame.
    uint32_t scissorHeight = plan.scissor_height;
    static const bool s_keepLastLine = std::getenv("DOL_GXCORE_KEEP_SCISSOR_479") != nullptr;
    if (!s_keepLastLine && plan.scissor_x == 0 && plan.scissor_y == 0 && plan.scissor_width == 640 &&
        plan.scissor_height == 479) {
      scissorHeight = 480;
    }
    gx::set_logical_scissor({
        .x = plan.scissor_x,
        .y = plan.scissor_y,
        .width = plan.scissor_width,
        .height = static_cast<int32_t>(scissorHeight),
    });
  }

  // The game's own shadow under a character or object: the triangles of the
  // ground under it drawn again, blended, in a texture of its shape (see
  // frame_interp's draw_key_of()). With the sun shadow map on, the object
  // casts its shadow from the map, and the game's darkened its feet a second
  // time, a dark patch inside the shade of a building.
  // DOL_AURORA_GAME_SHADOWS=1 keeps them.
  static const bool s_keepGameShadows = [] {
    const char* value = std::getenv("DOL_AURORA_GAME_SHADOWS");
    return value != nullptr && value[0] == '1';
  }();
  // DOL_AURORA_GAME_SHADOW_LOG=1 (local debug): every 120th frame, each blended
  // textured draw with what the two rules below test, to find a game shadow
  // they let through.
  static const bool s_shadowLog = std::getenv("DOL_AURORA_GAME_SHADOW_LOG") != nullptr;
  if (s_shadowLog && plan.tex_address != 0 && plan.pipeline.blend_enable != 0u &&
      !(plan.tex_width == 32 && plan.tex_height == 32)) {
    static uint64_t s_frame = 0;
    static uint32_t s_count = 0;
    const uint64_t frame = frame_interp::game_frame_number();
    if (frame != s_frame) {
      s_frame = frame;
      s_count = 0;
    }
    if (frame % 120u == 0u && s_count++ < 400u)
      std::fprintf(stderr,
                   "[game-shadow] frame %llu direct=%d base=%08X prim=%02X tag=%u proj32=%g tex=%08X %ux%u fmt=%u map=%d\n",
                   static_cast<unsigned long long>(frame), plan.match_direct_position ? 1 : 0,
                   plan.match_position_base, plan.match_primitive, static_cast<unsigned>(plan.draw_tag),
                   plan.constants.projection[3][2], plan.tex_address, plan.tex_width, plan.tex_height,
                   plan.tex_format, post_fx::sun_shadow_map_enabled() ? 1 : 0);
  }
  if (!s_keepGameShadows && plan.match_direct_position && plan.draw_tag == 0 && plan.match_primitive == 0x90 &&
      plan.tex_address != 0 && plan.pipeline.blend_enable != 0u && plan.constants.projection[3][2] != 0.f &&
      post_fx::sun_shadow_map_enabled())
    return true;
  // The quad that lays the grey composite's frame copy back while monotone (copy_efb_to_texture()): Wind Waker HD's
  // monotone replaces it. The first flat draw that samples the copy, once.
  if (g_monotoneDrop != 0 && plan.constants.projection[3][3] == 1.f && plan.pipeline.shader.textured != 0) {
    bool samples = plan.texmap_mask == 0u && plan.has_texture && (plan.tex_address & 0x3FFFFFFFu) == g_monotoneDrop;
    for (uint32_t t = 0; t < 8u && !samples; ++t)
      samples = (plan.texmap_mask & (1u << t)) != 0u && plan.textures[t].valid &&
                (plan.textures[t].address & 0x3FFFFFFFu) == g_monotoneDrop;
    if (samples) {
      g_monotoneDrop = 0;
      static bool s_logged = false;
      if (!s_logged) {
        s_logged = true;
        std::fprintf(stderr, "[monotone] the game's grey quad left out\n");
      }
      return true;
    }
  }
  // The rest of the game's own shadows: every one, real or simple (the blob
  // under a pot, a rupee or a character, and the shadow-volume passes),
  // indexes its positions from one of two fixed arrays (d_drawlist's
  // l_shadowVolPos and l_simpleShadowPos). The map casts the real shadows.
  // The base is compared without its top bits: on Linux it comes as the
  // physical address (0x0037C7DC), and the simple shadows got through.
  if (!s_keepGameShadows && !plan.match_direct_position &&
      ((plan.match_position_base & 0x3FFFFFFFu) == 0x0037C554u ||
       (plan.match_position_base & 0x3FFFFFFFu) == 0x0037C7DCu) &&
      post_fx::sun_shadow_map_enabled())
    return true;

  // Resolve one texmap's texture to a GPU handle: EFB-copy shadow first, else the
  // guest-identity + content-hash decode cache. Shared by the single-texmap fast
  // path and the multi-texmap path (63/Mfin, e.g. THP YUV Y/U/V on texmap 0/1/2).
  // A texture that is a copy of the EFB marks a flat draw as the frame laid back
  // over itself (post_fx goes in before it: common.cpp).
  bool samplesCopy = false;
  auto resolve_texture_handle =
      [&](uint32_t address, uint32_t tsize, uint32_t format, uint32_t width,
          uint32_t height, const void* data, uint32_t available, bool has_tlut,
          uint32_t tlut_address, uint32_t tlut_format, uint32_t tlut_entries,
          const void* tlut_data, uint32_t tlut_available) -> TextureHandle {
    // Only CI formats read a palette. The frontend reports whatever TLUT was
    // last bound for every texture, so without this a non-CI image drawn under
    // two different palettes had two cache identities at one address, evicted
    // itself and was decoded and uploaded again on every alternation: Wind
    // Waker's HUD did that about 23 times a frame (I4/IA4 images, same texels).
    if (!gxc::is_ci_format(format)) {
      has_tlut = false;
      tlut_address = 0u;
      tlut_format = 0u;
      tlut_entries = 0u;
      tlut_data = nullptr;
      tlut_available = 0u;
    }
    // The pack's replacements changed (an optional/ folder loaded or dropped as the HD look goes on or off): the
    // decode cache holds lookups made before, a replacement that is gone or a decode a new one replaces.
    static uint64_t s_replacementGeneration = texture_replacement::replacement_generation();
    if (const uint64_t generation = texture_replacement::replacement_generation();
        generation != s_replacementGeneration) {
      s_replacementGeneration = generation;
      g_textureCache.clear();
      g_textureAwaitingReplacement.clear();
      g_textureAddrKey.clear();
    }
    auto efbIt = g_efbCopyTextures.find(address);
    if (efbIt != g_efbCopyTextures.end() && efbIt->second.handle) {
      uint64_t memoryEpoch = 0;
      const bool memoryUnchanged =
          !efbIt->second.memory_epoch_valid ||
          (efb_copy_epoch(address, efbIt->second.byte_size, &memoryEpoch) &&
           memoryEpoch == efbIt->second.memory_epoch);
      if (memoryUnchanged) {
        ++g_textureCacheStats.hits;
        samplesCopy = true;
        return efbIt->second.handle;
      }
      g_efbCopyTextures.erase(efbIt);
    }
    if (data == nullptr)
      return TextureHandle{};
    const auto* bytes = static_cast<const uint8_t*>(data);
    const uint32_t size = std::min(tsize, available);
    uint64_t texelEpoch = 0;
    uint64_t tlutEpoch = 0;
    const bool texelEpochValid =
        g_textureDirtyEpochObserver != nullptr &&
        g_textureDirtyEpochObserver(address, size, &texelEpoch);
    const uint32_t tlutSize =
        gxc::is_ci_format(format) && has_tlut && tlut_data != nullptr
            ? std::min(tlut_available, tlut_entries * 2u)
            : 0u;
    const bool tlutEpochValid =
        tlutSize == 0u ||
        (g_textureDirtyEpochObserver != nullptr &&
         g_textureDirtyEpochObserver(tlut_address, tlutSize, &tlutEpoch));
    auto addrIt = g_textureAddrKey.find(address);
    if (texelEpochValid && tlutEpochValid && addrIt != g_textureAddrKey.end()) {
      const TextureAddressState& previous = addrIt->second;
      const TextureKey& priorKey = previous.key;
      const bool sameIdentity =
          priorKey.address == address && priorKey.size == tsize &&
          priorKey.format == format && priorKey.width == width &&
          priorKey.height == height && priorKey.tlut_address == tlut_address &&
          priorKey.tlut_format == tlut_format &&
          priorKey.tlut_entries == tlut_entries;
      const uint64_t frame = g_textureVerifyFrame.load(std::memory_order_relaxed);
      bool small_changed = false;
      if (sameIdentity && size <= kAlwaysRehashBytes && previous.verified_frame != frame) {
        const uint64_t h = small_texture_hash(bytes, size, tlut_data, tlutSize);
        if (h != previous.small_hash)
          small_changed = true;
        else
          addrIt->second.verified_frame = frame;
      }
      if (sameIdentity && !small_changed && previous.texel_epoch_valid &&
          previous.tlut_epoch_valid && previous.texel_epoch == texelEpoch &&
          previous.tlut_epoch == tlutEpoch) {
        const TextureHandle* cached = previous.handle ? &previous.handle : nullptr;
        if (cached == nullptr) {
          if (auto found = g_textureCache.find(priorKey); found != g_textureCache.end())
            cached = &found->second;
        }
        if (cached != nullptr) {
          // DOL_GXCORE_TEX_VERIFY=1: re-hash every generation hit and report the
          // ones whose texels changed without a dirty mark (a write path the
          // dirty tracking does not see).
          static const bool s_verify = std::getenv("DOL_GXCORE_TEX_VERIFY") != nullptr;
          if (s_verify) {
            const auto* vbytes = static_cast<const uint8_t*>(data);
            if (vbytes != nullptr &&
                XXH3_64bits(vbytes, std::min(tsize, available)) != priorKey.content_hash &&
                !gxc::is_ci_format(format))
              std::fprintf(stderr, "[tex-stale] addr=%08X fmt=%u %ux%u size=%u epoch=%llu\n",
                           address, format, width, height, tsize,
                           (unsigned long long)texelEpoch);
          }
          ++g_textureCacheStats.hits;
          ++g_textureCacheStats.generation_hits;
          return *cached;
        }
      }
    } else if (!texelEpochValid || !tlutEpochValid) {
      ++g_textureCacheStats.generation_fallbacks;
    }
    ++g_textureCacheStats.hashed_lookups;
    uint64_t content_hash = XXH3_64bits(bytes, size);
    if (gxc::is_ci_format(format) && has_tlut && tlut_data != nullptr) {
      ++g_textureCacheStats.palette_hashes;
      // Resolvers report bytes available to the end of their mapped range. Only
      // the declared GX palette belongs to this texture cache identity.
      content_hash = XXH3_64bits_withSeed(tlut_data, tlutSize, content_hash);
    }
    const uint64_t small_hash =
        size <= kAlwaysRehashBytes ? small_texture_hash(bytes, size, tlut_data, tlutSize) : 0u;
    const uint64_t verify_frame = size <= kAlwaysRehashBytes
                                      ? g_textureVerifyFrame.load(std::memory_order_relaxed)
                                      : ~0ull;
    const TextureKey key{address,     tsize,        format,      width,
                         height,      tlut_address, tlut_format, tlut_entries,
                         content_hash};
    auto it = g_textureCache.find(key);
    if (it != g_textureCache.end()) {
      ++g_textureCacheStats.hits;
      g_textureAddrKey[address] = TextureAddressState{
          key, texelEpoch, tlutEpoch, texelEpochValid, tlutEpochValid,
          small_hash, verify_frame, it->second};
      return it->second;
    }
    // New content for this identity: evict any prior entry cached at the same
    // guest address (its buffer was overwritten) to keep the cache bounded.
    addrIt = g_textureAddrKey.find(address);
    if (addrIt != g_textureAddrKey.end()) {
      g_textureCache.erase(addrIt->second.key);
      addrIt->second.handle = {};
    }
    // HD texture packs (Dolphin's tex1_WxH_hash[_tlut]_fmt names): the first
    // time these guest bytes are seen, look for a replacement before decoding.
    // The handle is cached under the same content key as a decode would be,
    // so later draws of the same texture never repeat the lookup.
    bool replacementPending = false;
    if (texture_replacement::has_source_replacements()) {
      static unsigned long long s_lookups;
      static const bool s_log_misses = std::getenv("DOL_TEXREP_LOG_MISSES") != nullptr;
      ++s_lookups;
      const auto replacement = texture_replacement::find_replacement_for_guest(
          width, height, format, bytes, size, static_cast<const uint8_t*>(tlut_data),
          tlut_data != nullptr ? std::min(tlut_available, tlut_entries * 2u) : 0u,
          &replacementPending);
      if (s_log_misses && !replacementPending && !(replacement.has_value() && *replacement))
        std::fprintf(stderr, "[mods] texture miss #%llu addr=%08X fmt=%u %ux%u size=%u tlut=%u\n", s_lookups,
                     address, format, width, height, size, tlut_entries);
      if (replacement.has_value() && *replacement) {
        g_textureAwaitingReplacement.erase(key);
        ++g_textureCacheStats.uploads;
        static unsigned long long s_replaced;
        if (++s_replaced <= 3u || (s_replaced % 200u) == 0u)
          std::fprintf(stderr, "[mods] texture replaced #%llu addr=%08X fmt=%u %ux%u -> %ux%u\n",
                       s_replaced, address, format, width, height, (*replacement)->size.width,
                       (*replacement)->size.height);
        g_textureCache.emplace(key, *replacement);
        g_textureAddrKey[address] = TextureAddressState{
            key, texelEpoch, tlutEpoch, texelEpochValid, tlutEpochValid,
            small_hash, verify_frame, *replacement};
        return *replacement;
      }
      if (replacementPending) {
        if (auto it = g_textureAwaitingReplacement.find(key); it != g_textureAwaitingReplacement.end())
          return it->second;
      } else {
        g_textureAwaitingReplacement.erase(key);
      }
    }
    // gxcore owns the decode for every format: produce tightly-packed RGBA8 and
    // upload it as a pre-decoded PC texture (no substrate re-conversion).
    std::vector<uint8_t> decoded;
    if (gxc::is_ci_format(format)) {
      if (has_tlut && tlut_data != nullptr)
        decoded = gxc::decode_ci(format, width, height, bytes, size, tlut_format,
                                 tlut_entries,
                                 static_cast<const uint8_t*>(tlut_data),
                                 tlut_available);
    } else {
      decoded = gxc::decode_texture(format, width, height, bytes, size);
    }
    {
      static const bool s_upload_log = std::getenv("DOL_GXCORE_TEX_UPLOAD_LOG") != nullptr;
      if (s_upload_log)
        std::fprintf(stderr, "[tex-upload] addr=%08X fmt=%u %ux%u size=%u efb=%d prior=%d tlut=%08X hash=%016llX pfmt=%u p%ux%u psize=%u\n",
                     address, format, width, height, tsize,
                     g_efbCopyTextures.find(address) != g_efbCopyTextures.end() ? 1 : 0,
                     addrIt != g_textureAddrKey.end() ? 1 : 0, tlut_address,
                     (unsigned long long)content_hash,
                     addrIt != g_textureAddrKey.end() ? addrIt->second.key.format : 0u,
                     addrIt != g_textureAddrKey.end() ? addrIt->second.key.width : 0u,
                     addrIt != g_textureAddrKey.end() ? addrIt->second.key.height : 0u,
                     addrIt != g_textureAddrKey.end() ? addrIt->second.key.size : 0u);
    }
    {
      // DOL_GXCORE_DUMP_TEX=<hex address>: write the decoded RGBA of that
      // texture as /tmp/gxcore-tex-<address>-<n>.pam (diagnostics).
      static const long long s_dump = [] {
        const char* env = std::getenv("DOL_GXCORE_DUMP_TEX");
        return env != nullptr ? std::strtoll(env, nullptr, 16) : -1ll;
      }();
      static int s_dumped = 0;
      if (s_dump >= 0 && (address & 0x3FFFFFFFu) == (static_cast<uint32_t>(s_dump) & 0x3FFFFFFFu) &&
          !decoded.empty() && s_dumped < 4) {
        char path[128];
        std::snprintf(path, sizeof(path), "/tmp/gxcore-tex-%08X-%d.pam", address, s_dumped++);
        if (FILE* f = std::fopen(path, "wb")) {
          std::fprintf(f, "P7\nWIDTH %u\nHEIGHT %u\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n", width, height);
          std::fwrite(decoded.data(), 1, decoded.size(), f);
          std::fclose(f);
        }
      }
    }
    TextureHandle handle;
    if (!decoded.empty()) {
      handle = new_static_texture_2d(
          width, height, 1, GX_TF_RGBA8_PC,
          ArrayRef<uint8_t>{decoded.data(), decoded.size()}, false,
          "GXCore Texture");
      ++g_textureCacheStats.uploads;
      if (gxc::is_ci_format(format))
        ++g_textureCacheStats.ci_uploads;
    } else {
      // CI without a resolved palette, or a format gxcore does not decode: upload
      // the raw GX bytes under the original format (old behavior).
      handle = new_static_texture_2d(width, height, 1, format,
                                     ArrayRef<uint8_t>{bytes, size}, false,
                                     "GXCore Texture");
      ++g_textureCacheStats.uploads;
      ++g_textureCacheStats.raw_fallback;
    }
    if (replacementPending) {
      g_textureAwaitingReplacement[key] = handle;
      return handle;
    }
    g_textureCache.emplace(key, handle);
    g_textureAddrKey[address] = TextureAddressState{
        key, texelEpoch, tlutEpoch, texelEpochValid, tlutEpochValid,
        small_hash, verify_frame, handle};
    return handle;
  };

  BindGroupRef textureBindGroup = 0;
  BindGroupRef cutoutBindGroup = 0; // DrawData::cutoutBindGroup
  if (plan.pipeline.shader.textured != 0) {
    if (plan.texmap_mask == 0u) {
      // Single-texmap fast path: primary texture at binding 0/1 (unchanged).
      if (plan.has_texture) {
        TextureHandle bound = resolve_texture_handle(
            plan.tex_address, plan.tex_size, plan.tex_format, plan.tex_width,
            plan.tex_height, plan.tex_data, plan.tex_available, plan.has_tlut,
            plan.tlut_address, plan.tlut_format, plan.tlut_entries,
            plan.tlut_data, plan.tlut_available);
        if (bound) {
          // Draws mostly bind the texture and sampler of the draw before. In
          // one frame (the bind group cache keeps what a frame used) the same
          // view and sampler state are the same bind group, without the
          // layout's, the sampler's and the bind group's hash and lock. The
          // generation changes when the renderer is set up again, whose frame
          // count starts over.
          struct LastBind {
            WGPUTextureView view = nullptr;
            gxc::PlanSampler sampler{};
            uint32_t frame = UINT32_MAX;
            uint32_t generation = 0;
            BindGroupRef ref = 0;
          };
          thread_local LastBind s_last;
          const gxc::PlanSampler& samplerState = plan.samplers[plan.tex_slot & 7u];
          const WGPUTextureView view = bound->sampleTextureView.Get();
          const uint32_t frame = current_frame();
          const uint32_t generation = pipeline_cache_generation();
          if (s_last.frame == frame && s_last.generation == generation && s_last.view == view &&
              std::memcmp(&s_last.sampler, &samplerState, sizeof samplerState) == 0) {
            textureBindGroup = s_last.ref;
          } else {
            const auto sampler = sampler_ref(sampler_descriptor(samplerState, replacement_mips(bound)));
            const std::array entries{
                WGPUBindGroupEntry{.binding = 0, .textureView = view},
                WGPUBindGroupEntry{.binding = 1, .sampler = sampler.Get()},
            };
            const WGPUBindGroupDescriptor descriptor{
                .label = {"GXCore Texture Bind Group", WGPU_STRLEN},
                .layout = texture_bind_group_layout(1u).Get(),
                .entryCount = entries.size(),
                .entries = entries.data(),
            };
            textureBindGroup = bind_group_ref(descriptor);
            s_last = LastBind{view, samplerState, frame, generation, textureBindGroup};
          }
          cutoutBindGroup = textureBindGroup;
        }
      }
    } else {
      // Multi-texmap: bind each used texmap at 2t (texture) / 2t+1 (sampler),
      // matching the WGSL declarations. If any referenced texmap fails to
      // resolve the draw is not drawable (avoids a missing-bind-group error).
      // At most eight texmaps, two entries each: fixed arrays, so a draw does
      // not allocate (the title screen issues about 20,000 draws a frame).
      std::array<TextureHandle, 8> held{};
      std::array<wgpu::Sampler, 8> heldSamplers{};
      std::array<WGPUBindGroupEntry, 16> entries{};
      // Each texmap's view and sampler, for the cutout's own group below.
      std::array<WGPUTextureView, 8> mapViews{};
      std::array<WGPUSampler, 8> mapSamplers{};
      size_t heldCount = 0;
      size_t entryCount = 0;
      bool complete = true;
      for (uint32_t t = 0; t < 8u; ++t) {
        if ((plan.texmap_mask & (1u << t)) == 0u)
          continue;
        const gxc::PlanTexture& pt = plan.textures[t];
        TextureHandle bound;
        if (pt.valid)
          bound = resolve_texture_handle(
              pt.address, pt.size, pt.format, pt.width, pt.height, pt.data,
              pt.available, pt.has_tlut, pt.tlut_address, pt.tlut_format,
              pt.tlut_entries, pt.tlut_data, pt.tlut_available);
        if (!bound) {
          complete = false;
          break;
        }
        held[heldCount] = bound;
        heldSamplers[heldCount] = sampler_ref(sampler_descriptor(plan.samplers[t], replacement_mips(bound)));
        entries[entryCount++] = WGPUBindGroupEntry{
            .binding = 2u * t, .textureView = bound->sampleTextureView.Get()};
        entries[entryCount++] = WGPUBindGroupEntry{
            .binding = 2u * t + 1u, .sampler = heldSamplers[heldCount].Get()};
        mapViews[t] = bound->sampleTextureView.Get();
        mapSamplers[t] = heldSamplers[heldCount].Get();
        ++heldCount;
      }
      if (!complete)
        return false;
      const WGPUBindGroupDescriptor descriptor{
          .label = {"GXCore Texture Bind Group", WGPU_STRLEN},
          .layout = texture_bind_group_layout(plan.texmap_mask).Get(),
          .entryCount = entryCount,
          .entries = entries.data(),
      };
      textureBindGroup = bind_group_ref(descriptor);
      // An alpha-tested draw of more than one texmap (a Windfall girl's body: its colour and alpha from texmap 0, its
      // toon ramp from texmap 1) casts in the sun map with the cutout of the texmap its alpha is read from, bound
      // alone as the single-texmap casters' texture is. Without it neither caster rule took the draw, and the
      // character cast no shadow (the game's own under her is left out while the map is on).
      if (sun_caster_alpha(plan.pipeline) || sun_caster_blended_cutout(plan.pipeline)) {
        const int cutoutMap = gxc::cutout_alpha_texmap(plan.pipeline.shader);
        if (cutoutMap >= 0 && mapViews[cutoutMap] != nullptr) {
          const std::array cutoutEntries{
              WGPUBindGroupEntry{.binding = 0, .textureView = mapViews[cutoutMap]},
              WGPUBindGroupEntry{.binding = 1, .sampler = mapSamplers[cutoutMap]},
          };
          const WGPUBindGroupDescriptor cutoutDescriptor{
              .label = {"GXCore Cutout Bind Group", WGPU_STRLEN},
              .layout = texture_bind_group_layout(1u).Get(),
              .entryCount = cutoutEntries.size(),
              .entries = cutoutEntries.data(),
          };
          cutoutBindGroup = bind_group_ref(cutoutDescriptor);
        }
      }
    }
  }

  // A cloth's strips (the boat's sail, a flag: a scope over draws that index
  // their positions) are solid fabric the game draws blended, with more than
  // one texmap for the sail's emblem: neither caster rule took them, so the
  // sail cast nothing. Their shape is their geometry, no cutout needed, so
  // they cast through the solid pipeline whenever they write depth.
  const bool clothCaster = plan.draw_scope_part != 0 && !plan.match_direct_position &&
                           plan.pipeline.depth_test != 0u && plan.pipeline.depth_update != 0u &&
                           plan.pipeline.color_update != 0u && plan.pipeline.primitive_topology == 0u;
  // Much of the game's solid furniture is drawn blended (a treasure chest,
  // Beedle's ship, a Wizrobe), which the solid caster rule refuses so the sea
  // and glows cast nothing. What tells them apart: the sea rewrites its vertex
  // grid every frame to follow the player, furniture never does. A blended,
  // depth-writing, indexed draw whose positions stood still casts.
  const bool blendedCaster = !plan.match_direct_position && plan.draw_scope_part == 0 &&
                             plan.pipeline.blend_enable != 0u && plan.pipeline.blend_subtract == 0u &&
                             plan.pipeline.depth_test != 0u && plan.pipeline.depth_update != 0u &&
                             plan.pipeline.color_update != 0u && plan.pipeline.primitive_topology == 0u &&
                             alpha_test_always_passes(plan.pipeline.shader) && plan.pipeline.shader.textured != 0u &&
                             !positions_written(plan);
  static const bool s_casterLog = [] {
    const char* value = std::getenv("DOL_AURORA_SUN_CASTER_LOG");
    return value != nullptr && value[0] == '1';
  }();
  if (s_casterLog && (plan.draw_scope_part != 0 ||
                      (plan.vertex_count >= 100 && !sun_caster(plan.pipeline) && !clothCaster && !blendedCaster &&
                       !sun_caster_alpha(plan.pipeline)))) {
    static std::set<uint64_t> seen;
    const uint64_t state = (uint64_t(plan.pipeline.blend_enable) << 0) | (uint64_t(plan.pipeline.depth_test) << 1) |
                           (uint64_t(plan.pipeline.depth_update) << 2) | (uint64_t(plan.pipeline.color_update) << 3) |
                           (uint64_t(plan.pipeline.primitive_topology) << 4) |
                           (uint64_t(alpha_test_always_passes(plan.pipeline.shader)) << 8) |
                           (uint64_t(plan.pipeline.shader.textured) << 9) | (uint64_t(plan.texmap_mask) << 16) |
                           (uint64_t(plan.tex_address) << 24);
    if (seen.insert(state).second)
      std::fprintf(stderr,
                   "[sun-caster] scope=%06X:%u tex=%08X blend=%u depth_test=%u depth_update=%u colour=%u topo=%u "
                   "alpha_always=%u textured=%u texmap_mask=%02X verts=%u -> solid=%d alpha=%d\n",
                   plan.draw_scope, plan.draw_scope_part, plan.tex_address, plan.pipeline.blend_enable,
                   plan.pipeline.depth_test, plan.pipeline.depth_update, plan.pipeline.color_update,
                   plan.pipeline.primitive_topology, alpha_test_always_passes(plan.pipeline.shader) ? 1u : 0u,
                   plan.pipeline.shader.textured, plan.texmap_mask, plan.vertex_count,
                   (sun_caster(plan.pipeline) || clothCaster) ? 1 : 0, sun_caster_alpha(plan.pipeline) ? 1 : 0);
  }
  const bool tev = plan.pipeline.shader.tev_valid != 0;
  const size_t vertBytes = plan.vertices.size() * sizeof(float);
  const size_t indexBytes = plan.indices.size() * sizeof(uint16_t);
  const size_t pixelUniformBytes = tev ? sizeof(plan.pixel_constants) : 0;
  // One comparison for the three de-duplications below (the vertex block, the
  // in-between frame's pool and its block): most draws repeat the constants of
  // the draw before them, 96 percent of the Forsaken Fortress's 17,500 a frame,
  // and each comparison of equal blocks reads all 2.8 KB of both.
  // A plan whose constants were kept from the draw before (constants_id, set
  // by GxCoreState::build_draw_plan_into) is known to repeat the block last
  // pushed in this frame packet without comparing them; constants made anew
  // are compared (they may still be the same bytes).
  const uint64_t frameId = current_frame_id();
  const bool repeatsLast =
      (plan.constants_id != 0 && frameId != 0 && plan.constants_id == g_pushedConstantsId &&
       g_vertexUniformCache.frameId == frameId && g_vertexUniformCache.range.size == sizeof(plan.constants)) ||
      repeats_cached(g_vertexUniformCache, reinterpret_cast<const uint8_t*>(&plan.constants), sizeof(plan.constants));
  // In-between frames: matched on the helper thread (queued below), or here
  // while a traced frame reports each draw's outcome. Here it is matched
  // before a staging segment can split the frame; a split frame is not
  // interpolated. A frame the pacing gave no in-between frames
  // (frame_skipped) queues nothing: matching its draws would only cost the
  // CPU it was dropped for, and the frame after has no previous to blend from
  // (as after a cut).
  const bool interpolating = frame_interp::enabled() && !frame_interp::frame_skipped();
  const bool matchHere = interpolating && frame_interp::tracing();
  const bool positionsWritten = interpolating && frame_interp::may_blend_vertices(plan) && positions_written(plan);
  if (matchHere)
    wait_interp_jobs();
  static frame_interp::DrawInput tracedInput;
  if (matchHere)
    frame_interp::capture_draw(plan, tracedInput, positionsWritten);
  const gxc::VertexShaderConstants* interpConstants =
      matchHere ? frame_interp::blend_draw(tracedInput, plan.constants, repeatsLast) : nullptr;
  // The helper is idle while a frame is traced, so its areas are ours.
  static std::vector<float> tracedVertices;
  const float* tracedPositions = matchHere ? frame_interp::blended_positions() : nullptr;
  const float* tracedTexcoords = matchHere ? frame_interp::blended_texcoords() : nullptr;
  const Range interpVertRange =
      tracedPositions != nullptr || tracedTexcoords != nullptr
          ? push_blended_vertices(recording_frame_slot(), plan.vertices, tracedPositions, tracedTexcoords,
                                  frame_interp::blended_texcoord_mask(), tracedVertices)
          : Range{};
  if (frame_interp::tracing()) {
    const auto& m = plan.constants.posnormalmatrix;
    std::fprintf(stderr,
                 "[frame-interp-trace] frame=%llu %s key=%016llx prim=0x%02X fmt=%u verts=%u payload=%u idx=%d "
                 "t=(%.1f,%.1f,%.1f) s=%.3f proj00=%.3f proj32=%.1f tex=%08X bt=(%.1f,%.1f,%.1f) bt0=(%.1f,%.1f,%.1f) "
                 "direct=%d tag=%06X age=%u scope=%06X:%u positions=%d uv_mask=%u uv0=(%.7f,%.7f) buv0=(%.7f,%.7f) "
                 "bp0=(%.3f,%.3f,%.3f)\n",
                 static_cast<unsigned long long>(frame_interp::game_frame_number()), frame_interp::last_outcome(),
                 static_cast<unsigned long long>(frame_interp::draw_key(plan)),
                 plan.match_primitive, plan.match_vtx_fmt, plan.vertex_count, plan.match_payload_size,
                 plan.pipeline.shader.has_pos_mtx_idx, m[0][3], m[1][3], m[2][3],
                 std::sqrt(m[0][0] * m[0][0] + m[1][0] * m[1][0] + m[2][0] * m[2][0]), plan.constants.projection[0][0],
                 plan.constants.projection[3][2], plan.tex_address,
                 interpConstants ? interpConstants->posnormalmatrix[0][3] : 0.f,
                 interpConstants ? interpConstants->posnormalmatrix[1][3] : 0.f,
                 interpConstants ? interpConstants->posnormalmatrix[2][3] : 0.f,
                 interpConstants ? interpConstants->transformmatrices[0][3] - plan.constants.transformmatrices[0][3] : 0.f,
                 interpConstants ? interpConstants->transformmatrices[1][3] - plan.constants.transformmatrices[1][3] : 0.f,
                 interpConstants ? interpConstants->transformmatrices[2][3] - plan.constants.transformmatrices[2][3] : 0.f,
                 plan.match_direct_position ? 1 : 0, plan.draw_tag, plan.draw_tag_age, plan.draw_scope, plan.draw_scope_part,
                 tracedPositions != nullptr ? 1 : 0, frame_interp::blended_texcoord_mask(),
                 plan.vertices.empty() ? 0.f : plan.vertices[gxc::kVertexUvOffset / sizeof(float)],
                 plan.vertices.empty() ? 0.f : plan.vertices[gxc::kVertexUvOffset / sizeof(float) + 1],
                 tracedTexcoords != nullptr ? tracedTexcoords[0] : 0.f,
                 tracedTexcoords != nullptr ? tracedTexcoords[1] : 0.f,
                 tracedPositions != nullptr ? tracedPositions[0] : 0.f,
                 tracedPositions != nullptr ? tracedPositions[1] : 0.f,
                 tracedPositions != nullptr ? tracedPositions[2] : 0.f);
  }
  // (Room too for a pixel block of the ubershader's, should the draw need one.)
  const size_t pixelRoom = pixelUniformBytes + sizeof(gxc::UberPixelConstants);
  if (!staging_has_capacity(vertBytes + gxc::kVertexStrideBytes, indexBytes, sizeof(plan.constants), pixelRoom)) {
    if (!segment_frame() ||
        !staging_has_capacity(vertBytes + gxc::kVertexStrideBytes, indexBytes, sizeof(plan.constants), pixelRoom)) {
      Log.error("GXCore draw exceeds an empty Aurora staging segment");
      return false;
    }
  }

  const auto uniformRange = push_uniform_dedup(
      g_vertexUniformCache, reinterpret_cast<const uint8_t*>(&plan.constants),
      sizeof(plan.constants), repeatsLast ? 1 : 0);
  g_pushedConstantsId = plan.constants_id;
  const bool seaSurface = sea_mask_draw(plan);
  // Still scenery the game blends and alpha-tests (its grass and bushes, sun_caster_blended_cutout()) casts with its
  // texture's cutout, as Wind Waker HD casts its grass.
  const bool blendedCutout = !plan.match_direct_position && plan.draw_scope_part == 0 &&
                             plan.constants.projection[3][3] != 1.f && sun_caster_blended_cutout(plan.pipeline) &&
                             !character_draw(plan.pipeline) && !seaSurface && !positions_written(plan);
  const bool alphaCaster = cutoutBindGroup != 0 && (sun_caster_alpha(plan.pipeline) || blendedCutout);
  const bool smoothNormal = smooth_normal_draw(plan, seaSurface);
  // A palm with Wind Waker HD's light (apply_hd_colours()): its shade colour black. A lava draw with the GameCube's
  // own glow (hd_lava_gc_glow()) that takes HD's (t_hdLavaDraw below, the same tests): C1 black too.
  const bool lightDraw = smoothNormal && hd_light_draw(plan);
  const bool noGcGlow = lightDraw && tev && hd_lava_gc_glow(plan.pipeline.shader) &&
                        post_fx::point_lights_hd_switch() && post_fx::actor_light_hd_switch() &&
                        gxc::hd_point_light_stage(plan.pipeline.shader) && hd_lava_draw(plan);
  const gxc::PixelShaderConstants& lightConstants =
      lightDraw ? hd_light_constants(plan, noGcGlow) : plan.pixel_constants;
  // Wind Waker HD's fog start before the camera (aurora_set_fog_hd_starts) on every draw fogged from the game's
  // clamped start (C 0) to the same end: the characters, plants and objects as well as the lit scenery, all fogged
  // from that start in HD's shaders. The lit scenery's fog the sun pass takes (note_scene_fog()) is these constants.
  const gxc::PixelShaderConstants& pixelConstants = tev ? hd_fog_constants(lightConstants, plan) : lightConstants;
  Range pixelUniformRange{};
  bool pixelRepeats = false;
  if (tev) {
    const uint64_t hits = g_pixelUniformCache.hits;
    pixelUniformRange = push_uniform_dedup(
        g_pixelUniformCache,
        reinterpret_cast<const uint8_t*>(&pixelConstants),
        sizeof(pixelConstants));
    pixelRepeats = g_pixelUniformCache.hits != hits;
  }
  // A lightning strike (dThunder_c) draws, with its bolt, an invisible model
  // that writes depth only, about 100,000 units out: a strip from the sea to
  // the clouds. The HD look's effects read the depth, so the far blur and the
  // cloud shadows took that strip of sky for something solid and drew it as a
  // grey column. A depth-only draw that far out writes none while they are on.
  gxc::PipelineKey key = plan.pipeline;
  if (key.depth_update != 0u && key.color_update == 0u && key.alpha_update == 0u && post_fx::any_enabled() &&
      plan.constants.projection[3][3] != 1.f) {
    const auto& m = plan.constants.posnormalmatrix;
    if (m[0][3] * m[0][3] + m[1][3] * m[1][3] + m[2][3] * m[2][3] > 50000.f * 50000.f)
      key.depth_update = 0u;
  }
  // Wind Waker HD's character light (DOL_AURORA_ACTOR_LIGHT_HD=1): a toon draw in the ramp pattern
  // (gxc::toon_light_stage()) takes the flagged shader, HD's light beside the GX result, picked per pass.
  // DOL_AURORA_ACTOR_LIGHT_HD_EXT=0 (to compare): only the main pattern, not the ones after it (E, F, G: the
  // colour-register albedo, an added register, later stages; ToonLightMatch::after_albedo).
  static const bool extended = [] {
    const char* env = std::getenv("DOL_AURORA_ACTOR_LIGHT_HD_EXT");
    return env == nullptr || env[0] != '0';
  }();
  if (tev && post_fx::actor_light_hd_switch()) {
    const gxc::ToonLightMatch toon = gxc::toon_light_stage(key.shader);
    if (toon.matched && (extended || !toon.after_albedo))
      key.shader.hd_actor_light = 1u;
  }
  // Wind Waker HD's light buffer on the lit scenery (DOL_AURORA_POINT_LIGHTS_HD=1, with its character light): a draw
  // whose stage 0 is the GameCube's light (gxc::hd_point_light_stage()) takes the flagged shader, which adds HD's
  // lights x 1.25 to it while the pass has them (the characters' group 0; none in the off group). 2: its vertices
  // carry the smooth normals made for the sun shadows (with_smooth_normals()), which it takes for the lights' N.L.
  if (tev && post_fx::point_lights_hd_switch() && post_fx::actor_light_hd_switch() &&
      key.shader.hd_actor_light == 0u && gxc::hd_point_light_stage(key.shader))
    key.shader.hd_point_light = smoothNormal ? 2u : 1u;
  // Wind Waker HD's lava glow (hd_lava_draw()), in the lit scenery's flagged shader (4) and its vertices' colour 1.
  t_hdLavaDraw = smoothNormal && key.shader.hd_point_light != 0u && hd_lava_draw(plan);
  if (t_hdLavaDraw)
    key.shader.hd_point_light |= 4u;
  // Wind Waker HD's water floor (gxc::hd_water_floor_stage()), its sum in linear light on HD's linear colours, while
  // HD's palettes (whose colours it takes) and its light are in place.
  if (tev && gxc::hd_water_floor_stage(key.shader) && post_fx::actor_light_wanted())
    key.shader.hd_water = 1u;
  // Wind Waker HD's fog curve (aurora_set_fog_hd): while the host has HD's fog distances in the game's palettes,
  // linear fog is squared, as HD's shaders fog; on the GameCube's linear curve HD's distances (retuned for the
  // square) gave about twice HD's fog at mid range.
  if (tev && key.shader.fog_fsel != static_cast<uint8_t>(gxc::FogType::Off) && post_fx::fog_hd())
    key.shader.hd_fog = 1u;
  // The lit scenery's fog (its linear perspective fog, with GX's range adjust if it has it), for the sun pass to put after
  // its shade, as Wind Waker HD fogs its scenery (post_fx::note_scene_fog()).
  if (tev && key.shader.fog_fsel == static_cast<uint8_t>(gxc::FogType::Linear) &&
      key.shader.fog_proj == static_cast<uint8_t>(gxc::FogProjection::Perspective) && hd_light_draw(plan)) {
    const auto& pc = pixelConstants;
    post_fx::note_scene_fog(pc.fogf[0], pc.fogi[1], pc.fogi[3], pc.fogf[1], pc.fogcolor, key.shader.hd_fog != 0u,
                            key.shader.fog_range != 0u ? &pc.fogrange[0][0] : nullptr, pc.fogf[2], pc.fogf[3]);
  }
  const PipelineConfig colorConfig{
      .version = GXCorePipelineConfigVersion,
      .key = key,
      .msaaSamples = get_sample_count(),
  };
  const PipelineRef pipeline = pipeline_ref(colorConfig);
  PipelineRef depthPipeline = 0;
  if (needs_early_depth_emulation(key)) {
    PipelineConfig depthConfig = colorConfig;
    depthConfig.depthOnly = 1u;
    depthPipeline = pipeline_ref(depthConfig);
  }
  const bool ownVertices = plan.draw_tag != 0 || plan.draw_scope_part != 0 ||
                           (interpolating && frame_interp::blends_vertices(plan, positionsWritten));
  // Efficient soft effects: its variant for the reduced-size buffer, looked up
  // (and compiled) only while the setting is on.
  const bool softEffect = efficient_effects() && soft_effects::qualifies_key(plan.pipeline);
  PipelineRef softPipeline = 0;
  if (softEffect) {
    PipelineConfig softConfig = colorConfig;
    softConfig.msaaSamples = 1;
    softConfig.softTarget = 1;
    softPipeline = pipeline_ref(softConfig);
  }

  // The ubershader (gxcore_uber.cpp): a draw whose own pipeline is still
  // compiling is drawn with it, with the same result, rather than left out of
  // the frame (Dolphin's ubershaders). Its pipelines are one per fixed-function
  // state, few, and kept in the pipeline cache like any other. The draw gets
  // a pixel block with its shader key and a bind group with all eight texmaps,
  // used only if its own pipeline is still not ready when it is encoded.
  // On by default on D3D12 (ubershader_mode()); DOL_AURORA_UBERSHADER=0 turns
  // it off, =2 draws every draw with it (testing: a frame drawn both ways can
  // be compared).
  static const int uberMode = ubershader_mode();
  PipelineRef uberPipeline = 0;
  Range uberPixelRange{};
  BindGroupRef uberTextureBindGroup = 0;
  if (uberMode != 0 && (uberMode == 2 || !pipeline_ready(pipeline) ||
                        (depthPipeline != 0 && !pipeline_ready(depthPipeline)))) {
    // All eight texmap slots: on the single-texmap path the draw's texture in
    // every one (as its own shader samples slot 0 whatever the stage's texmap
    // says), else each used texmap where it is bound; the rest empty.
    const wgpu::TextureView& empty = empty_texmap_view();
    const auto emptySampler = sampler_ref(sampler_descriptor(gxc::PlanSampler{}));
    std::array<WGPUTextureView, 8> views{};
    std::array<wgpu::Sampler, 8> samplers{};
    views.fill(empty.Get());
    samplers.fill(emptySampler);
    std::array<TextureHandle, 8> held{};
    bool complete = true;
    if (plan.pipeline.shader.textured != 0) {
      if (plan.texmap_mask == 0u) {
        if (plan.has_texture)
          held[0] = resolve_texture_handle(plan.tex_address, plan.tex_size, plan.tex_format, plan.tex_width,
                                           plan.tex_height, plan.tex_data, plan.tex_available, plan.has_tlut,
                                           plan.tlut_address, plan.tlut_format, plan.tlut_entries, plan.tlut_data,
                                           plan.tlut_available);
        complete = static_cast<bool>(held[0]);
        if (complete) {
          views.fill(held[0]->sampleTextureView.Get());
          samplers.fill(sampler_ref(sampler_descriptor(plan.samplers[plan.tex_slot & 7u], replacement_mips(held[0]))));
        }
      } else {
        for (uint32_t t = 0; t < 8u && complete; ++t) {
          if ((plan.texmap_mask & (1u << t)) == 0u)
            continue;
          const gxc::PlanTexture& pt = plan.textures[t];
          if (pt.valid)
            held[t] = resolve_texture_handle(pt.address, pt.size, pt.format, pt.width, pt.height, pt.data,
                                             pt.available, pt.has_tlut, pt.tlut_address, pt.tlut_format,
                                             pt.tlut_entries, pt.tlut_data, pt.tlut_available);
          complete = static_cast<bool>(held[t]);
          if (complete) {
            views[t] = held[t]->sampleTextureView.Get();
            samplers[t] = sampler_ref(sampler_descriptor(plan.samplers[t], replacement_mips(held[t])));
          }
        }
      }
    }
    if (complete) {
      std::array<WGPUBindGroupEntry, 16> entries{};
      for (uint32_t t = 0; t < 8u; ++t) {
        entries[2u * t] = WGPUBindGroupEntry{.binding = 2u * t, .textureView = views[t]};
        entries[2u * t + 1u] = WGPUBindGroupEntry{.binding = 2u * t + 1u, .sampler = samplers[t].Get()};
      }
      const WGPUBindGroupDescriptor descriptor{
          .label = {"GXCore Ubershader Texture Bind Group", WGPU_STRLEN},
          .layout = texture_bind_group_layout(0xFFu).Get(),
          .entryCount = entries.size(),
          .entries = entries.data(),
      };
      uberTextureBindGroup = bind_group_ref(descriptor);
      PipelineConfig uberConfig = colorConfig;
      uberConfig.key.shader = gxc::ShaderKey{};
      uberConfig.key.shader.use_dst_alpha = plan.pipeline.shader.use_dst_alpha;
      uberConfig.depthOnly = 2u;
      uberPipeline = pipeline_ref(uberConfig);
      static gxc::UberPixelConstants block; // one recording thread at a time
      std::memcpy(&block.psc, &pixelConstants, sizeof block.psc);
      std::memset(block.key, 0, sizeof block.key);
      std::memcpy(block.key, &key.shader, sizeof key.shader); // with the per-draw bits (hd_fog)
      block.extra[0] = gxc::texmap_popcount(gxc::used_texmap_mask(plan.pipeline.shader)) > 1u ? 1u : 0u;
      uberPixelRange = push_uniform_dedup(g_uberPixelUniformCache, reinterpret_cast<const uint8_t*>(&block),
                                          sizeof block);
    }
  }

  // Batching: a draw of the state of the pass's last command (pipeline,
  // constants, pixel constants, textures), with nothing between them, whose
  // vertices and indices follow its, extends it, its indices counted from the
  // batch's first vertex. Grass, rocks, a particle system's quads and a wake's
  // strips are runs of such draws: 65 percent of Adanmae's 10,000 draws a
  // frame repeat the constants of the draw before. In-between frames draw a
  // batch as one where its draws' blocks came out the same (render()).
  static const bool batching = [] {
    const char* env = std::getenv("DOL_AURORA_GXCORE_BATCH");
    return env == nullptr || env[0] != '0';
  }();
  const size_t vertexOffset = next_vertex_offset(gxc::kVertexStrideBytes);
  DrawData* batch = nullptr;
  uint32_t firstVertex = 0;
  // Independently blended vertices cannot share a batch's rebased index
  // range. Keep their original draw order, including each depth/color pair.
  if (batching && !matchHere && !ownVertices) {
    DrawData* last = last_recorded_draw();
    if (last != nullptr && uberPipeline == 0 && last->uberPipeline == 0 && last->pipeline == pipeline &&
        last->depthPipeline == depthPipeline &&
        last->uniformRange.offset == uniformRange.offset && last->tev == tev &&
        (!tev || last->pixelUniformRange.offset == pixelUniformRange.offset) &&
        last->textureBindGroup == textureBindGroup && last->ownVertices == ownVertices &&
        last->smoothNormal == smoothNormal && last->farCaster == plan.far_caster &&
        last->interpUniformRange.size == 0 && last->interpVertRange.size == 0 &&
        (interpolating ? last->interpJob != UINT32_MAX && last->interpJob + last->batchSize == next_interp_job()
                       : last->interpJob == UINT32_MAX) &&
        last->vertRange.offset % gxc::kVertexStrideBytes == 0 &&
        last->vertRange.offset + last->vertRange.size == vertexOffset &&
        last->idxRange.offset + last->idxRange.size == next_index_offset()) {
      const size_t first = (vertexOffset - last->vertRange.offset) / gxc::kVertexStrideBytes;
      if (first + plan.vertex_count <= 65536u) {
        batch = last;
        firstVertex = static_cast<uint32_t>(first);
      }
    }
  }

  // On a whole vertex: render() draws it at a base vertex of the whole buffer. Scenery without normals takes
  // smooth ones while the facing term is on (with_smooth_normals()).
  const auto vertRange = push_verts_strided(
      reinterpret_cast<const uint8_t*>(smoothNormal ? with_smooth_normals(plan).data() : plan.vertices.data()),
      vertBytes, gxc::kVertexStrideBytes);
  t_hdLavaDraw = false;
  Range idxRange;
  if (batch != nullptr) {
    static std::vector<uint16_t> rebased;
    rebased.resize(plan.indices.size());
    for (size_t i = 0; i < rebased.size(); ++i)
      rebased[i] = static_cast<uint16_t>(plan.indices[i] + firstVertex);
    idxRange = push_indices(reinterpret_cast<const uint8_t*>(rebased.data()), indexBytes, 2);
  } else {
    idxRange = push_indices(reinterpret_cast<const uint8_t*>(plan.indices.data()), indexBytes, 2);
  }
  Range interpUniformRange{};
  uint32_t interpJob = UINT32_MAX;
  if (interpConstants != nullptr)
    interpUniformRange = push_interp_uniform_dedup(
        g_interpUniformCache, current_frame_id(), recording_frame_slot(), reinterpret_cast<const uint8_t*>(interpConstants),
        sizeof(*interpConstants), frame_interp::last_blend_repeated());
  else if (interpolating && !matchHere)
    interpJob = queue_interp_job(plan, pixelConstants, repeatsLast, tev, pixelRepeats, positionsWritten);

  const auto indexCount = static_cast<uint32_t>(plan.indices.size());
  if (batch != nullptr) {
    if (batch->batchSize == 1)
      batch->batch = push_batch_draw({.indexCount = batch->indexCount, .firstVertex = 0});
    push_batch_draw({.indexCount = indexCount, .firstVertex = firstVertex});
    ++batch->batchSize;
    batch->indexCount += indexCount;
    batch->idxRange.size += idxRange.size;
    batch->vertRange.size = static_cast<uint32_t>(vertRange.offset + vertRange.size - batch->vertRange.offset);
    ++g_mergedDrawCallCount;
    return true;
  }
  set_next_draw_flat(plan.constants.projection[3][3] == 1.f);
  set_next_draw_composite(plan.constants.projection[3][3] == 1.f && samplesCopy);
  if (plan.constants.projection[3][3] != 1.f && plan.pipeline.depth_update != 0u && !character_draw(plan.pipeline))
    log_terrain_normals(plan.pipeline, plan.has_texture ? plan.tex_address : plan.textures[0].address, plan.texmap_mask);
  // What writes depth in the frame but is in none of the sea mask's other kinds: blended scenery that writes depth
  // (Outset's bushes: alpha-tested, and blended to fade as the camera nears). The sun map leaves it out (blended),
  // but it hides what is behind it in the frame, so the sea mask's pass draws it too (render_sea_mask()). Without
  // it, the marks of what stood behind (Link's character and HD light, a palm's green and normals) showed through
  // the bush in the sun shadows. Only a shape the mask can draw: whole (no alpha test) or by its one texture's
  // cutout. A card cut out by more textures, drawn whole, would hide the marks of what shows through its holes.
  // An effect that writes depth (effect_draw()) is left out: the sea mask's depth keeps the scenery's under it.
  const bool effectDepth = effect_depth_draw(plan);
  // A particle drawn alpha-tested into the depth alone and then in colour over it (Dragon Roost Cavern's dust: the
  // emitter's depth draws first, then their colour draws): its colour draw, an effect that writes no depth, is drawn
  // at its own depth into the effects' mask too, where pushed back it failed against the depth draw's depth.
  static uint64_t s_depthParticlesFrame = UINT64_MAX;
  static absl::flat_hash_set<uint32_t> s_depthParticles;
  if (plan.draw_tag != 0u) {
    if (s_depthParticlesFrame != frame_interp::game_frame_number()) {
      s_depthParticlesFrame = frame_interp::game_frame_number();
      s_depthParticles.clear();
    }
    if (plan.pipeline.depth_update != 0u && plan.pipeline.color_update == 0u)
      s_depthParticles.insert(plan.draw_tag);
  }
  const bool effectOwnDepth = effectDepth || (plan.draw_tag != 0u && plan.pipeline.depth_test != 0u &&
                                              s_depthParticles.contains(plan.draw_tag) && effect_draw(plan));
  const bool maskOccluderShape =
      alpha_test_always_passes(plan.pipeline.shader) ||
      (plan.texmap_mask == 0u && textureBindGroup != 0 && plan.pipeline.shader.textured != 0u);
  const bool maskOccluder = plan.pipeline.depth_test != 0u && plan.pipeline.depth_update != 0u &&
                            plan.pipeline.color_update != 0u && plan.pipeline.primitive_topology == 0u &&
                            plan.constants.projection[3][3] != 1.f && maskOccluderShape &&
                            !alpha_test_always_fails(plan.pipeline.shader) &&
                            !((sun_caster(plan.pipeline) || clothCaster || blendedCaster) ||
                              alphaCaster || character_draw(plan.pipeline) || seaSurface || effectDepth);
  push_draw_command(DrawData{
      .pipeline = pipeline,
      .depthPipeline = depthPipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .uniformRange = uniformRange,
      .interpUniformRange = interpUniformRange,
      .interpVertRange = interpVertRange,
      .interpJob = interpJob,
      .pixelUniformRange = pixelUniformRange,
      .indexCount = indexCount,
      .textureBindGroup = textureBindGroup,
      .cutoutBindGroup = cutoutBindGroup,
      .tev = tev,
      .ownVertices = ownVertices,
      .uberPipeline = uberPipeline,
      .uberPixelRange = uberPixelRange,
      .uberTextureBindGroup = uberTextureBindGroup,
      .sunCaster = (sun_caster(plan.pipeline) || clothCaster || blendedCaster) &&
                   !(sun_caster(plan.pipeline) && post_fx::sun_shadow_map_enabled() && sea_surface(plan)),
      .posMtxIdx = plan.pipeline.shader.has_pos_mtx_idx != 0,
      .sunCasterCull = plan.pipeline.cull_mode,
      .farCaster = plan.far_caster,
      // DOL_AURORA_SUN_CASTER_LOG=1: why a cloth's or wake's draw (a scope)
      // does or does not cast, once per distinct state (local debug).

      // The cutout pipeline binds one texture and sampler at group 2, the
      // single-texmap layout: a multi-texmap material keeps the screen's steps.
      .sunCasterAlpha = alphaCaster,
      .seaSurface = seaSurface,
      .depthWrite = plan.pipeline.depth_test != 0u && plan.pipeline.depth_update != 0u,
      .softEffect = softEffect,
      .softWritesAlpha = plan.pipeline.alpha_update != 0,
      .softPipeline = softPipeline,
      .effect = effect_draw(plan),
      .effectDepth = effectDepth,
      .effectOwnDepth = effectOwnDepth,
      .character = character_draw(plan.pipeline),
      .smoothNormal = smoothNormal,
      .maskCover = plan.pipeline.depth_update == 0u && plan.pipeline.color_update != 0u &&
                   plan.pipeline.blend_enable == 0u && plan.pipeline.blend_subtract == 0u &&
                   plan.pipeline.primitive_topology == 0u,
      .hasNormal = (plan.pipeline.shader.has_vertex_normal != 0u && plan.pipeline.shader.raw_tex_hi_in_nbt == 0u) ||
                   smoothNormal,
      .toon = toon_draw(plan.pipeline),
      .actorLight = key.shader.hd_actor_light != 0u,
      .pointLight = key.shader.hd_point_light != 0u,
      .maskOccluder = maskOccluder,
      .maskOccluderCutout = maskOccluder && !alpha_test_always_passes(plan.pipeline.shader),
      .maskOccluderLit = maskOccluder && plan.pipeline.shader.lit_valid != 0u,
  });
  return true;
}

} // namespace aurora::gfx::gxcore

void aurora_set_efficient_effects(bool on) {
  aurora::gfx::gxcore::g_efficientEffects.store(on, std::memory_order_relaxed);
}

void aurora_set_hd_normals(const char* path) {
  using namespace aurora::gfx::gxcore;
  std::shared_ptr<const HdNormals> table = path != nullptr && path[0] != '\0' ? load_hd_normals(path) : nullptr;
  std::lock_guard lock{g_hdNormalsMutex};
  g_hdNormals = std::move(table);
}

void aurora_set_hd_stage_colours(const char* path) {
  using namespace aurora::gfx::gxcore;
  std::shared_ptr<const HdColours> table = path != nullptr && path[0] != '\0' ? load_hd_colours(path, true) : nullptr;
  std::lock_guard lock{g_hdNormalsMutex};
  g_hdStageColours = std::move(table);
}

void aurora_set_hd_object_tables(const char* normals_path, const char* colours_path) {
  using namespace aurora::gfx::gxcore;
  std::shared_ptr<const HdNormals> normals =
      normals_path != nullptr && normals_path[0] != '\0' ? load_hd_normals(normals_path) : nullptr;
  std::shared_ptr<const HdColours> colours =
      colours_path != nullptr && colours_path[0] != '\0' ? load_hd_colours(colours_path) : nullptr;
  std::lock_guard lock{g_hdNormalsMutex};
  g_hdObjectNormals = std::move(normals);
  g_hdObjectColours = std::move(colours);
}

void aurora_set_forced_anisotropy(unsigned samples) {
  aurora::gfx::gxcore::g_forcedAnisotropy.store(
      static_cast<std::uint16_t>(std::clamp(samples, 1u, 16u)), std::memory_order_relaxed);
}
