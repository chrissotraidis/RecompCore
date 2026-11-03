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
// distance as brightness. _HD=1: Wind Waker HD's linear ramp in place of the
// smoothstep, clamp(level x (distance - start) / (end - start)) x the amount,
// the level the host's for the hour (aurora_set_far_blur_level).
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
// Torch light: the game's point lights (aurora_set_point_lights: up to 32 of
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
void set_far_blur_level(float level) noexcept;
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
// The shadow quality, 0 low, 1 medium, 2 high, 3 Wind Waker HD's own shadow pass
// (DOL_AURORA_SUN_SHADOW_QUALITY's levels, "hd" for 3); applied by the render thread before its next frame.
void set_sun_shadow_quality(int level) noexcept;
void set_sun_shadow_strength(float strength) noexcept;
void set_sun_cascade_splits(float middle, float far) noexcept;
void set_sun_palette(bool on, const float sky[3], const float ground[3], const float lit[3]) noexcept;
// Wind Waker HD's character light (gxcore_shader.cpp, ShaderKey::hd_actor_light): DOL_AURORA_ACTOR_LIGHT_HD=1 (read
// once) lets the toon characters' draws take it; the host gives the hour's colours (aurora_set_actor_light_hd,
// linear). actor_light_palette(): the colours, false while none are given (the GameCube shading).
bool actor_light_hd_switch() noexcept;
// Wind Waker HD's light buffer (DOL_AURORA_POINT_LIGHTS_HD=1, read once; off unless set): the point lights the host
// gives (aurora_set_point_lights) are on HD's rule (radius 2.3 x power, or the effect light's power; colour S10 / 255,
// unclamped), and the draws with HD's light add HD's sum of them, colour x sat(N.L) x sat(1 - d / r)^1.5: the
// characters' x 1.75 (gxc::HdPointLights), with HD's character light. point_lights_hd_active(): on, the torch light on
// (the host sends the lights) and the characters' light wanted this frame.
bool point_lights_hd_switch() noexcept;
bool point_lights_hd_active() noexcept;
// Wind Waker HD's fog curve (aurora_set_fog_hd): linear fog squared (gxc::ShaderKey::hd_fog), set by the host while
// HD's fog distances are in the game's palettes. Read per draw.
bool fog_hd() noexcept;
void set_fog_hd(bool on) noexcept;
// Wind Waker HD's palette fog starts below 0, as start, end pairs (aurora_set_fog_hd_starts).
void set_fog_hd_starts(const float* start_end, int count) noexcept;
// The fog the lit scenery's draws took this frame (GX's: A, B's magnitude and shift, C, its colour 0-255, and HD's
// square), for the sun pass to fog after the shade, by the distance from the eye, as Wind Waker HD's scenery
// shaders do (DOL_AURORA_SUN_FOG_HD=0: as drawn). From the draw thread.
void note_scene_fog(float a, int b_magnitude, int b_shift, float c, const int colour[3], bool hd_square,
                    const float* range = nullptr, float centre = 0.f, float width = 0.f) noexcept;
void set_actor_light_hd(bool on, const float c0[3], const float shadow[3], const float k[3]) noexcept;
bool actor_light_palette(float c0[3], float shadow[3], float k[3]) noexcept;
// The GameCube's global actor C0 and K0 (linear) the characters' own registers are set against
// (aurora_set_actor_light_hd_own); false while none are given or with DOL_AURORA_ACTOR_LIGHT_HD_OWN=0.
void set_actor_light_hd_own(bool on, const float c0[3], const float k[3]) noexcept;
bool actor_light_own(float c0[3], float k[3]) noexcept;
// The switch is on and the host has given colours: the characters may take HD's light this frame.
bool actor_light_wanted() noexcept;
// The shadow the characters' light reads: the last post pass's blurred shadow pre-pass (HD's terrain mix or its
// shadow pass, made after the scene, so last frame's), and its LOD to read (HD's mip 1 with its shadow pass). False
// when the last post pass made none.
// With the characters' own pre-pass made for this pass (draw_actor_shadow()), that one (this frame's, mip 1) and
// `sameFrame` true.
bool actor_shadow_source(wgpu::TextureView& view, float& lod, bool& sameFrame) noexcept;
// DOL_AURORA_ACTOR_LIGHT_SAME_FRAME (read once; on unless 0): the characters' shadow from this frame, from a
// pre-pass of their own (draw_actor_shadow()); 0 takes last frame's pre-pass alone (for A/B).
bool actor_light_same_frame() noexcept;
// This frame's scene pass drew the characters with HD's light (gxcore_draw.cpp, begin_actor_light_pass()): the
// effects after it leave the toon-shaded pixels (the sea mask's green over 0.8) to it (the sun shadows' shade,
// HD's rim light, the brightness), since HD's light already has them.
void set_actor_light_applied(bool applied) noexcept;
bool actor_light_applied() noexcept;
// Wind Waker HD's cloud shadow texture (aurora_set_hd_cloud_texture): a BC4 DDS, loaded by the render thread; null
// or empty: none (BlueWake's noise).
void set_hd_cloud_texture(const char* path) noexcept;
// The characters' cloud term (Wind Waker HD's character light, gxc::HdActorLightConstants::cloud): the scenery's
// cloud shadows on HD's rule as fs_sun_map applies them, from the same values (the texture and its repeating
// sampler, the repeat, the drift from camera.view[16]'s seconds, v0 0.5, the floor uniform[87] has). `cloud` gets
// (u0, v0, 1 / repeat, floor). False, and the characters' cloud 1, while the cloud shadows on HD's rule are off this
// frame or no HD cloud texture is loaded (the scenery's noise fallback isn't HD's, so characters don't take it).
struct Camera;
bool actor_cloud_source(const Camera& camera, float cloud[4], wgpu::TextureView& view, wgpu::Sampler& sampler) noexcept;
// Wind Waker HD's light rule (aurora_set_sun_hd_light).
void set_sun_hd_light(bool on, bool night, float fade) noexcept;
// Wind Waker HD's cloud floor from the host's clock (aurora_set_cloud_hd_floor), negative while unset.
void set_cloud_hd_floor(float floor) noexcept;
// The scenery's shadow casters culled as their draws are (aurora_set_sun_caster_cull; DOL_AURORA_SUN_SHADOW_CASTER_CULL
// =1 or 0 overrides it).
void set_sun_caster_cull(bool on) noexcept;
bool sun_caster_cull() noexcept;
void set_sun_direction(float x, float y, float z) noexcept;
void set_exposure(float gain) noexcept;
// The scene is brightened this frame (the game's gain over 1): the sea's draws
// are then marked for the sea mask (gxcore_draw.hpp render_sea_mask()).
bool exposure_active() noexcept;
// The sea mask (below) is wanted this frame: the scene is brightened, or the sun shadow map's facing term
// needs it to leave characters and the sea out (it's on with no brightening too: at 10:00 on Outset the
// game's gain isn't over 1).
bool sea_mask_wanted() noexcept;
// Wind Waker HD's bloom mask is wanted (DOL_AURORA_BLOOM_HD_MASK, with bloom on): the sea mask's pass also draws
// what writes no depth (the sky) and marks in blue what may glow (gxcore_draw.hpp render_sea_mask()).
bool bloom_mask_wanted() noexcept;
// Wind Waker HD's rim light on characters is on (DOL_AURORA_RIM_HD): the sea mask's pass draws the characters'
// own normals into sun_normal_target() (mask_normals_wanted()).
bool rim_hd_active() noexcept;
// The sea mask's pass draws normals beside it: the scenery's smooth ones for the facing term
// (sun_normals_wanted()), or the characters' for the rim light.
bool mask_normals_wanted() noexcept;
// The sea mask the brightness leaves the sea out by: red 1 where the sea is the
// nearest surface, green 1 where a character is (a solid skinned draw: the sun
// shadows' facing term leaves characters to their own shading), from the
// frame's solid draws and the sea drawn again into it and a depth of its own
// (gxcore_draw.hpp render_sea_mask()), the scene's size and single-sampled.
// common.cpp clears both at a frame's first scene pass and adds each scene
// pass, before encode().
// Blue: 0.5 where Wind Waker HD's bloom may start (bloom_mask_wanted()), 0 elsewhere.
constexpr wgpu::TextureFormat kSeaMaskFormat = wgpu::TextureFormat::RGBA8Unorm;
constexpr wgpu::TextureFormat kSeaMaskDepthFormat = wgpu::TextureFormat::Depth32Float;
wgpu::TextureView sea_mask_target(const wgpu::Extent3D& sceneSize);
wgpu::TextureView sea_mask_depth(const wgpu::Extent3D& sceneSize);
// The effects' mask (common.cpp encode_effect_mask()): what the frame's effects
// (gxcore_draw.hpp DrawData::effect) added to each pixel, drawn again with their
// own pipelines over black, in the frame buffer's format, at the scene's size.
// The sun shadows, the occlusion and the cloud shadows leave as much of a pixel
// as it holds unshaded. Asking for it marks it drawn for the next encode().
wgpu::TextureView effect_mask_target(const wgpu::Extent3D& sceneSize, wgpu::TextureFormat format);
// How far back the effects are drawn into it, a share of their distance
// (DOL_AURORA_EFFECT_MASK_PUSH, 0.05): what lies on a surface is left out.
float effect_mask_push() noexcept;
// The sun shadows' facing term is on: the scenery's smooth normals are wanted (made on the CPU for a draw
// without normals of its own, gxcore_draw.cpp, and drawn by the sea mask's pass into sun_normal_target()).
// Off, none of that work is done.
bool sun_normals_wanted() noexcept;
// The scenery's smooth normals beside the sea mask, the same size: the view-space normal x 0.5 + 0.5, alpha 1
// where a draw gave one; the facing term reads the depth's normal elsewhere.
constexpr wgpu::TextureFormat kSunNormalFormat = wgpu::TextureFormat::RGBA8Unorm;
wgpu::TextureView sun_normal_target(const wgpu::Extent3D& sceneSize);
bool cloud_shadows_enabled() noexcept;
void set_cloud_shadows(bool enabled) noexcept;
void set_world_view(const float invView[12], float tanHalfFovy, float aspect, float nearPlane, float seaLevel,
                    float seconds) noexcept;
void set_world_view_far(float farPlane) noexcept;
bool any_enabled() noexcept;
// The game's point lights, at most (aurora_set_point_lights; 16 before Wind Waker HD's light buffer).
constexpr uint32_t kMaxLights = 32;
// What the effects drawn from the camera (sun and cloud shadows, edge and
// torch light) are drawn with: the inverse view matrix's three rows, then
// tan(fovy / 2), aspect, near, the sea's height and the time in seconds, then
// the direction to the sun; and the point lights.
struct Camera {
  std::array<float, 20> view{};
  float farPlane = 0.f;  // 0: infinitely far (aurora_set_world_view_far)
  float exposure = 0.f;  // aurora_set_exposure; 0: unset
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
  // The light the map is drawn from: the sun, or at night the moon (opposite
  // it), and how strongly its shadows show (0 to 1; less for the moon, fading
  // out as it reaches the horizon).
  std::array<float, 3> light{};
  float fade = 0.f;
  bool valid = false;
};
SunMap sun_map(const Camera& camera) noexcept;
// The characters' shadow pre-pass for the scene pass about to be encoded (common.cpp render(), after the sea mask
// is drawn for it, before the pass): Wind Waker HD's shadow pre-pass from the sea mask's depth, the map's shadow at
// half the size blurred (mip 0), and mip 1 (a blurred eighth-size copy brought back up), with no blend with the
// last frames'. False when it can't be made (the camera or the map not valid, the mask of another size): the flagged
// draws then take last frame's (actor_shadow_source()). end_actor_shadow() after the pass.
bool draw_actor_shadow(const wgpu::CommandEncoder& cmd, const Camera& camera, const SunMap& sunMap,
                       const wgpu::Extent3D& size, wgpu::TextureFormat format);
void end_actor_shadow() noexcept;
// The map's depth texture (one layer a cascade), made at its size on first
// use: all of it, for the lookups.
const wgpu::TextureView& sun_map_view();
// One cascade's layer, for drawing its casters into.
const wgpu::TextureView& sun_map_layer(uint32_t cascade);
// Wind Waker HD's monotone (Frozen Hyrule's sepia, aurora_set_monotone_hd()): the host's parameters for its DOF
// pass, taken when a frame is recorded and kept on the frame's passes, so every in-between frame draws the same.
struct Monotone {
  bool on = false;
  float level = 0.8f;                     // how much of the sepia replaces the scene (HD's DOF level)
  float start = 0.f;                      // distance where it starts, nearer: colour (HD's DOF start and end)
  float end = 0.f;
  std::array<float, 3> mul{1.4f, 1.5f, 0.6f}; // the tint on the grey (HD's DOF colour multiply)
  float saturate = 1.f;                   // how far to the grey (1: all of it)
  uint32_t frameCopy = 0;                 // the game's frame copy (mDoGph_gInf_c::mFrameBufferTex), physical
};
// The host's monotone, while the HD look's version of it can be drawn (single-sampled, core features,
// DOL_AURORA_MONOTONE_HD not 0); else off, and the game draws its own grey.
Monotone monotone_params() noexcept;
void set_monotone(bool on, float level, float start, float end, const float mul[3], float saturate,
                  uint32_t frameCopy) noexcept;
bool monotone_hd_supported() noexcept;
// Which of the effects encode() draws: all of them; in a monotone frame, at the game's grey composite, the scene's
// (no far blur, as HD; no bloom) and then the monotone; or, after the player's lists, only the lighting (no far blur,
// no bloom, no monotone), for encode_late().
enum class Phase : uint8_t { Full, Scene, LateLight };
// Every effect that is on, in order (ambient occlusion, sun shadows, edge
// light, torch light, cloud shadows, far blur, bloom);
// depthView: the EFB's depth; camera: the one the frame is drawn from;
// sunMap: the shadow map drawn for this frame, or null (the screen's steps).
void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
            const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
            const wgpu::TextureView& resolve, uint32_t msaaSamples, const Camera& camera,
            const SunMap* sunMap = nullptr, Phase phase = Phase::Full, const Monotone* monotone = nullptr);
// A monotone frame's EFB depth at the grey composite (before the player's lists), for encode_late().
void snapshot_monotone_depth(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView,
                             const wgpu::Extent3D& size);
// A monotone frame after the player's lists (HD's pass 6, then its bloom): with `lateLight`, the lighting
// (Phase::LateLight) on a copy of the EFB, laid back where something was drawn nearer than the snapshot's depth (an
// arrow, a bomb, a rupee: what the game draws after the grey and Wind Waker HD's light doesn't light in its own
// draw); then bloom over all of it.
void encode_late(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
                 const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
                 const wgpu::TextureView& resolve, uint32_t msaaSamples, const Camera& camera,
                 const SunMap* sunMap, bool lateLight);
// scene: the EFB as a texture (resolved when multisampled), size: its size;
// dst: the EFB's color attachment, with msaaSamples samples, and resolve its
// single-sample copy when multisampled (null otherwise).
void encode_bloom(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::Extent3D& size,
                  wgpu::TextureFormat format, const wgpu::TextureView& dst, const wgpu::TextureView& resolve,
                  uint32_t msaaSamples);
} // namespace aurora::gfx::post_fx
