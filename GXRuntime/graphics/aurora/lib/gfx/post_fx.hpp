#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>

// Effects drawn after the game's own frame (the Wind Waker HD look).
//
// Bloom: once a frame, on the finished scene before the HUD (common.cpp's
// render), what is brighter than a threshold is blurred at a quarter of the
// size and added back. Off unless DOL_AURORA_BLOOM=1 or aurora_set_bloom(true);
// DOL_AURORA_BLOOM_THRESHOLD (0 to 1, of the brightness, default 0.72) and
// DOL_AURORA_BLOOM_STRENGTH (default 0.35) are read at startup.
//
// Far blur (HD's depth of field): before bloom, the scene blurred at a quarter
// of the size and blended back by distance, from DOL_AURORA_FAR_BLUR_START to
// _END (distances in near-plane units, default 8000 and 30000) up to
// DOL_AURORA_FAR_BLUR_AMOUNT (default 0.8); where the game drew without depth
// (the sky, the horizon, the islands' far models), _BACKGROUND (default 0.35). Off unless
// DOL_AURORA_FAR_BLUR=1 or aurora_set_far_blur(true); _DEBUG=1 shows it in red.
namespace aurora::gfx::post_fx {
void initialize();
void shutdown();
bool bloom_enabled() noexcept;
void set_bloom(bool enabled) noexcept;
bool far_blur_enabled() noexcept;
void set_far_blur(bool enabled) noexcept;
bool any_enabled() noexcept;
// Every effect that is on, in order (far blur, then bloom); depthView: the EFB's depth.
void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
            const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
            const wgpu::TextureView& resolve, uint32_t msaaSamples);
// scene: the EFB as a texture (resolved when multisampled), size: its size;
// dst: the EFB's color attachment, with msaaSamples samples, and resolve its
// single-sample copy when multisampled (null otherwise).
void encode_bloom(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::Extent3D& size,
                  wgpu::TextureFormat format, const wgpu::TextureView& dst, const wgpu::TextureView& resolve,
                  uint32_t msaaSamples);
} // namespace aurora::gfx::post_fx
