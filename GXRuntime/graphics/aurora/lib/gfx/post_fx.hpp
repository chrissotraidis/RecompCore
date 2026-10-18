#pragma once

#include <webgpu/webgpu_cpp.h>

#include <array>
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
//
// Torch light: the game's point lights (aurora_set_point_lights: up to 16 of
// x, y, z, radius, r, g, b, flicker) glow on the surfaces around them.
// DOL_AURORA_TORCH_LIGHT_STRENGTH (default 0.35) and _REACH (the glow's radius,
// times the game's lighting radius, default 3). Off unless
// aurora_set_torch_light(true).
namespace aurora::gfx::post_fx {
void initialize();
void shutdown();
bool bloom_enabled() noexcept;
void set_bloom(bool enabled) noexcept;
bool far_blur_enabled() noexcept;
void set_far_blur(bool enabled) noexcept;
bool ambient_occlusion_enabled() noexcept;
void set_ambient_occlusion(bool enabled) noexcept;
bool torch_light_enabled() noexcept;
void set_torch_light(bool enabled) noexcept;
void set_point_lights(const float* lights, uint32_t count) noexcept;
bool edge_light_enabled() noexcept;
void set_edge_light(bool enabled) noexcept;
bool sun_shadows_enabled() noexcept;
void set_sun_shadows(bool enabled) noexcept;
// Sun shadows on, and drawn from the shadow map.
bool sun_shadow_map_enabled() noexcept;
void set_sun_shadow_map(bool enabled) noexcept;
void set_sun_direction(float x, float y, float z) noexcept;
bool cloud_shadows_enabled() noexcept;
void set_cloud_shadows(bool enabled) noexcept;
void set_world_view(const float invView[12], float tanHalfFovy, float aspect, float nearPlane, float seaLevel,
                    float seconds) noexcept;
bool any_enabled() noexcept;
// The game's point lights, at most (aurora_set_point_lights).
constexpr uint32_t kMaxLights = 16;
// What the effects drawn from the camera (sun and cloud shadows, edge and
// torch light) are drawn with: the inverse view matrix's three rows, then
// tan(fovy / 2), aspect, near, the sea's height and the time in seconds, then
// the direction to the sun; and the point lights.
struct Camera {
  std::array<float, 20> view{};
  bool valid = false;
  std::array<float, kMaxLights * 8> lights{};
  uint32_t lightCount = 0;
};
// The camera the game set last (aurora_set_world_view and the rest).
Camera current_camera() noexcept;
// `camera` moved for an in-between frame: its inverse view times `relative`
// (the step's camera relative to the real frame's, frame_interp::camera_steps),
// at `seconds`.
Camera moved_camera(const Camera& camera, const float relative[3][4], float seconds) noexcept;
// The sun shadow map's depth format, and where a frame's maps are drawn from.
// The map is a few cascades, each wider than the one before (the first
// DOL_AURORA_SUN_SHADOW_MAP_RANGE across, each next _SPREAD times that), so
// what is near is sharp and an island far ahead still casts. For each cascade,
// `world` takes a place in the world to it (three rows of a 3x4 matrix: x and y
// -1 to 1 across it, z 0 to 1, nearer the sun less), `caster` takes the
// camera's view space there (world after the camera's inverse view), for the
// draws' positions. Not valid with the camera unknown or the sun too low.
constexpr wgpu::TextureFormat kSunMapFormat = wgpu::TextureFormat::Depth32Float;
constexpr uint32_t kSunMapCascades = 4;
struct SunMap {
  std::array<std::array<float, 12>, kSunMapCascades> world{};
  std::array<std::array<float, 12>, kSunMapCascades> caster{};
  uint32_t cascades = 0;
  bool valid = false;
};
SunMap sun_map(const Camera& camera) noexcept;
// The map's depth texture (one layer a cascade), made at its size on first
// use: all of it, for the lookups.
const wgpu::TextureView& sun_map_view();
// One cascade's layer, for drawing its casters into.
const wgpu::TextureView& sun_map_layer(uint32_t cascade);
// Every effect that is on, in order (ambient occlusion, sun shadows, edge
// light, torch light, cloud shadows, far blur, bloom);
// depthView: the EFB's depth; camera: the one the frame is drawn from;
// sunMap: the shadow map drawn for this frame, or null (the screen's steps).
void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
            const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
            const wgpu::TextureView& resolve, uint32_t msaaSamples, const Camera& camera,
            const SunMap* sunMap = nullptr);
// scene: the EFB as a texture (resolved when multisampled), size: its size;
// dst: the EFB's color attachment, with msaaSamples samples, and resolve its
// single-sample copy when multisampled (null otherwise).
void encode_bloom(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::Extent3D& size,
                  wgpu::TextureFormat format, const wgpu::TextureView& dst, const wgpu::TextureView& resolve,
                  uint32_t msaaSamples);
} // namespace aurora::gfx::post_fx
