#pragma once

#include "common.hpp"

#include <dolphin/gx/GXEnum.h>

namespace aurora::gfx::tex_copy_conv {

enum class SampleFilter : uint8_t {
  Nearest,
  Linear,
};

struct ConvRequest {
  GXTexFmt fmt;
  wgpu::TextureView srcView; // View of resolved EFB / offscreen color/depth
  Range uniformRange;        // UV transform uniform (offset + scale)
  TextureHandle dst;         // Destination texture
  SampleFilter sampleFilter = SampleFilter::Nearest;
  bool keepColour = false;   // GX_TF_I8 only: the colour beside the intensity (Wind Waker HD's monotone)
};

bool needs_conversion(GXTexFmt fmt);

void initialize();
void shutdown();
void run(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
void blit(const wgpu::CommandEncoder& cmd, const ConvRequest& req);
// Blur a colour copy in place: a Gaussian of `sigma` of its pixels, across then
// down, and `mix` of the blurred result over the sharp copy (1 replaces it).
void blur(const wgpu::CommandEncoder& cmd, const TextureHandle& texture, float sigma, float mix);
// The fast blur (aurora_set_game_dof_fast; DOL_AURORA_GAME_DOF_FAST): for a
// sigma of about 1 texel, blur() takes the copy down to half size and blends
// that back over the copy in one pass, instead of two full-size passes and a
// copy back. Off by default.
void set_fast_blur(bool on);
bool fast_blur();

} // namespace aurora::gfx::tex_copy_conv
