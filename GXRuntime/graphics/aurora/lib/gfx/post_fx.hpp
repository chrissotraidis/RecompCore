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
namespace aurora::gfx::post_fx {
void initialize();
void shutdown();
bool bloom_enabled() noexcept;
void set_bloom(bool enabled) noexcept;
// scene: the EFB as a texture (resolved when multisampled), size: its size;
// dst: the EFB's color attachment, with msaaSamples samples, and resolve its
// single-sample copy when multisampled (null otherwise).
void encode_bloom(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::Extent3D& size,
                  wgpu::TextureFormat format, const wgpu::TextureView& dst, const wgpu::TextureView& resolve,
                  uint32_t msaaSamples);
} // namespace aurora::gfx::post_fx
