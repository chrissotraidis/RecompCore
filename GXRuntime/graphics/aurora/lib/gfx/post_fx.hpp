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
// DOL_AURORA_FAR_BLUR=1 or aurora_set_far_blur(true); _DEBUG=1 shows the
// distance as brightness.
//
// Ambient occlusion (HD's darkened creases and contact shadows): first, from
// depth alone, at a quarter of the size, blurred and multiplied into the
// scene. DOL_AURORA_AO_RADIUS (near-plane units, default 30) and _STRENGTH
// (default 0.6). Off unless DOL_AURORA_AO=1 or aurora_set_ambient_occlusion(true).
//
// Cloud shadows: soft drifting shadows on the sea and the land, from the
// camera the game passes in (aurora_set_world_view) and the scene's depth.
// DOL_AURORA_CLOUD_SHADOWS_STRENGTH (default 0.28) and _SCALE (world units
// across a cloud, default 7000). Off unless aurora_set_cloud_shadows(true).
//
// Sun shadows: from what is on screen, toward the direction the game passes in
// (aurora_set_sun_direction), with the same camera. DOL_AURORA_SUN_SHADOWS_STRENGTH
// (default 0.35) and _REACH (world units, default 1500). Off unless
// aurora_set_sun_shadows(true).
//
// Edge light: a warm rim on the sunlit outline of what is near the camera,
// toward the same sun. DOL_AURORA_EDGE_LIGHT_STRENGTH (default 0.4). Off unless
// aurora_set_edge_light(true).
namespace aurora::gfx::post_fx {
void initialize();
void shutdown();
bool bloom_enabled() noexcept;
void set_bloom(bool enabled) noexcept;
bool far_blur_enabled() noexcept;
void set_far_blur(bool enabled) noexcept;
bool ambient_occlusion_enabled() noexcept;
void set_ambient_occlusion(bool enabled) noexcept;
bool edge_light_enabled() noexcept;
void set_edge_light(bool enabled) noexcept;
bool sun_shadows_enabled() noexcept;
void set_sun_shadows(bool enabled) noexcept;
void set_sun_direction(float x, float y, float z) noexcept;
bool cloud_shadows_enabled() noexcept;
void set_cloud_shadows(bool enabled) noexcept;
void set_world_view(const float invView[12], float tanHalfFovy, float aspect, float nearPlane, float seaLevel,
                    float seconds) noexcept;
bool any_enabled() noexcept;
// Every effect that is on, in order (ambient occlusion, sun shadows, edge
// light, cloud shadows, far blur, bloom);
// depthView: the EFB's depth.
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
