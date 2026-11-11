#ifndef AURORA_AURORA_H
#define AURORA_AURORA_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

extern "C" {
#else
#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"
#endif

typedef enum {
  SAMPLER_BILINEAR,
  SAMPLER_AREA,
} AuroraSampler;

typedef enum {
  BACKEND_AUTO,
  BACKEND_D3D11,
  BACKEND_D3D12,
  BACKEND_METAL,
  BACKEND_VULKAN,
  BACKEND_OPENGL,
  BACKEND_OPENGLES,
  BACKEND_WEBGPU,
  BACKEND_NULL,
} AuroraBackend;

typedef enum {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_FATAL,
} AuroraLogLevel;

typedef enum {
  FPS_OVERLAY_TOP_CENTER,
  FPS_OVERLAY_TOP_LEFT,
  FPS_OVERLAY_TOP_RIGHT,
  FPS_OVERLAY_BOTTOM_LEFT,
  FPS_OVERLAY_BOTTOM_RIGHT,
} AuroraFpsOverlayPosition;

typedef struct {
  int32_t x;
  int32_t y;
} AuroraWindowPos;

typedef struct {
  uint32_t width;
  uint32_t height;

  /**
   * Width of the main GX framebuffer.
   */
  uint32_t fb_width;

  /**
   * Height of the main GX framebuffer.
   */
  uint32_t fb_height;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_width if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_width;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_height if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_height;
  float scale;
} AuroraWindowSize;

typedef struct SDL_Window SDL_Window;
typedef struct AuroraEvent AuroraEvent;

typedef void (*AuroraLogCallback)(AuroraLogLevel level, const char* module, const char* message, unsigned int len);
typedef void (*AuroraImGuiInitCallback)(const AuroraWindowSize* size);

#define MEM1_DEFAULT_SIZE = 24 * 1024 * 1024;
#define ARAM_DEFAULT_SIZE = 16 * 1024 * 1024;

typedef struct {
  const char* appName;
  const char* userPath;
  const char* cachePath;
  const char* resourcesPath;
  AuroraBackend desiredBackend;
  uint32_t msaa;
  uint16_t maxTextureAnisotropy;
  bool vsync;
  bool startFullscreen;
  bool allowJoystickBackgroundEvents;
  bool pauseOnFocusLost;
  bool allowTextureDumps;
  bool allowCpuAdapter;
  int32_t windowPosX;
  int32_t windowPosY;
  uint32_t windowWidth;
  uint32_t windowHeight;
  void* iconRGBA8;
  uint32_t iconWidth;
  uint32_t iconHeight;
  AuroraLogCallback logCallback;
  AuroraLogLevel logLevel;
  AuroraImGuiInitCallback imGuiInitCallback;

  /*
   * The size of the GameCube's main memory, or MEM1 on the Wii.
   * Note that it will not be allocated at the exact 0x80000000 address, as that cannot be guaranteed.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem1Size;

  /*
   * The size of the GameCube's ARAM, or MEM2 on the Wii.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem2Size;
} AuroraConfig;

typedef struct {
  AuroraBackend backend;
  const char* userPath;
  const char* cachePath;
  SDL_Window* window;
  AuroraWindowSize windowSize;
} AuroraInfo;

AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config);
void aurora_shutdown();
const AuroraEvent* aurora_update();
bool aurora_begin_frame();
void aurora_end_frame();

void aurora_set_log_level(AuroraLogLevel level);
void aurora_set_pause_on_focus_lost(bool value);
void aurora_set_background_input(bool value);
void aurora_set_resampler(AuroraSampler sampler);
/* Internal render resolution as a multiple of the game's 640x480 frame
   buffer (3 renders 1920x1440 and scales the result to the window); 0 renders
   at the window's native pixel size. Takes effect at the next frame. */
void aurora_set_frame_buffer_scale(float scale);
/* While set, frames are rendered but not presented: no drawable is taken, so
   nothing waits for the display. The screen keeps the last presented frame. */
void aurora_set_present_suppressed(bool suppressed);
/* Anisotropic filtering for every mipmapped, linearly filtered texture, as
   Dolphin's "Anisotropic Filtering" enhancement: 1 (or 0) leaves the game's
   own sampler settings, 2-16 forces that many samples. Takes effect at the
   next draw. DOL_AURORA_FORCE_ANISO sets the starting value. */
void aurora_set_forced_anisotropy(unsigned samples);
/* Efficient soft effects: runs of large see-through quads drawn after the
   scene (haze, smoke, fog, clouds) are drawn into a buffer of half the EFB's
   size and blended over it once, at render scales of 2 and up. Off by default;
   takes effect for draws recorded after the change. */
void aurora_set_efficient_effects(bool on);
/* In-between frames: each finished frame is drawn a second time with every
   draw's transforms blended halfway toward the previous frame and shown
   first, so a 30 FPS game presents 60 frames a second with its own logic
   unchanged (one extra half frame of display latency). Takes effect at the
   next frame. DOL_AURORA_FRAME_INTERP=1 sets the starting value. */
void aurora_set_frame_interpolation(bool enabled);
/* Bloom (the Wind Waker HD look): what is brighter than a threshold glows,
   after the scene and before the HUD. Takes effect at the next frame.
   DOL_AURORA_BLOOM=1 sets the starting value. */
void aurora_set_bloom(bool enabled);
/* HD's far-distance blur: the distance blurred, the near scene and the sky
   sharp, before bloom. Takes effect at the next frame. DOL_AURORA_FAR_BLUR=1
   sets the starting value. */
void aurora_set_far_blur(bool enabled);
/* Wind Waker HD's far blur level for the hour (its DOF presets: wwHD_basic 1.2 by day, wwHD_night 0.65, blended
   over 06:00-08:00 and 18:00-20:00), used with DOL_AURORA_FAR_BLUR_HD=1: the blend is clamp(level x (distance -
   start) / (end - start)) times DOL_AURORA_FAR_BLUR_AMOUNT. 1.2 until set. Takes effect at the next frame. */
void aurora_set_far_blur_level(float level);
/* Ambient occlusion: creases and contact areas darkened, from the scene's
   depth, before the far blur and bloom. Takes effect at the next frame.
   DOL_AURORA_AO=1 sets the starting value. */
void aurora_set_ambient_occlusion(bool enabled);
/* Sun shadows from what is on screen, toward the direction last passed to
   aurora_set_sun_direction (world space, toward the sun), with the camera last
   passed to aurora_set_world_view. Takes effect at the next frame. */
void aurora_set_sun_shadows(bool enabled);
void aurora_set_sun_direction(float x, float y, float z);
/* The scene's brightness, from the game (Wind Waker HD's factor for the hour):
   what is lit and stands above the sea is multiplied by it, before bloom. 0
   (or 1) leaves the scene as it is. */
void aurora_set_exposure(float gain);
/* Sun shadows from a shadow map: with sun shadows on, the scene's solid draws
   are drawn again from the sun, so what is off screen still casts. Takes
   effect at the next frame. DOL_AURORA_SUN_SHADOW_MAP=1 sets the starting
   value. */
void aurora_set_sun_shadow_map(bool enabled);
/* The sun shadow map's quality: 0 low, 1 medium, 2 high (the levels of
   DOL_AURORA_SUN_SHADOW_QUALITY, which sets the starting one). The map is made
   again at its size before the next frame, a one-frame hitch. */
void aurora_set_sun_shadow_quality(int quality);
/* How dark the sun shadow map's shadows are, 0 to 1 (1, the default, is Wind
   Waker HD's): the full shadow's colour is blended in by this much, the
   moon's shadows too. Takes effect at the next frame. */
void aurora_set_sun_shadow_strength(float strength);
/* Wind Waker HD's shadow cascade splits for this frame, from the game's camera (HD sets them each frame from the
   camera's distance to what it looks at and its field of view): the distances from the camera where the "HD" shadow
   quality moves to its second and third cascade, whose first two cascades are sized with them. 0 goes back to the
   fixed ones (420 and 1395, HD's at Outset's bridge). Takes effect at the next frame. */
void aurora_set_sun_cascade_splits(float middle, float far);
/* Wind Waker HD's light for the sun shadow map's shade, the hour's (linear RGB): `sky`, the light from the sky
   its shade takes from above (its palette's shadow colour), `ground`, the ambient it takes from the side (the
   scenery's C0 times its factor), and `lit`, the sun's light (the scenery's K0 times its factor). A shaded pixel
   is multiplied by (their ambient over `lit`) in the scene's gamma. `on` 0 goes back to the fixed shadow colour.
   Takes effect at the next frame. */
void aurora_set_sun_palette(int on, const float sky[3], const float ground[3], const float lit[3]);
/* Wind Waker HD's character light for the hour (linear RGB): `c0`, the hemisphere's lower colour (actor_c0 times
   its factor), `shadow`, its upper colour (the palette's shadow colour), and `k`, the lit colour (actor_k0 times its
   factor: 2.75 at noon). Toon-shaded characters then take HD's light in place of the GameCube's ramp, when the
   renderer's switch DOL_AURORA_ACTOR_LIGHT_HD=1 is on (read at startup). `on` 0 goes back to the GameCube shading.
   Takes effect at the next frame. */
void aurora_set_actor_light_hd(int on, const float c0[3], const float shadow[3], const float k[3]);
/* Each character's own palettes on Wind Waker HD's light (its setLightTevColorType: a character's colours are its
   own tevstr's, from the floor under it, times the intensities of the global light, Link's): `c0` and `k0`, the
   GameCube's global actor C0 and K0 (g_env_light's, linear: bytes / 255 to the 2.2). A character draw whose own toon
   registers (C0 and KONST) differ from them takes HD's colours from aurora_set_actor_light_hd times the ratio, in
   linear light: an islander in the sun keeps the sun's light while Link stands under a roof whose floor names a
   darker palette. `on` 0: every character on the global colours. */
void aurora_set_actor_light_hd_own(int on, const float c0[3], const float k0[3]);
/* Wind Waker HD's light rule for the sun shadow map (its setLightPosHD): with `on`, the direction given by
   aurora_set_sun_direction is HD's light already (the sun by day, the moon's side by night, its height held as
   HD holds it), `night` says which, and `fade` (0 to 1) is HD's day/night switch fade, which takes the lit side
   down to the shade before the light changes sides. `on` 0 goes back to BlueWake's own rule. */
void aurora_set_sun_hd_light(int on, int night, float fade);
/* Wind Waker HD's cloud shadow floor (cCloudShadowPower, cloud = clamp(texel + floor)) from the game's clock, as HD
   works it out in setSunpos: 1 - max(0.25, -0.9 cos(time - 15 degrees)), 0.364 at 10:00, 0.131 at noon, 0.75 from
   a low sun through the night. Used by the cloud shadows on HD's rule, on the scenery and the characters. A negative
   value, the default, goes back to the renderer's own (from the shadow light's height), which is HD's only for a
   true sun: with HD's light rule the light is held at 40,000 high at dawn and dusk. DOL_AURORA_CLOUD_SHADOWS_HD_POWER
   still overrides both. Takes effect at the next frame. */
void aurora_set_cloud_hd_floor(float floor);
/* Wind Waker HD's caster culling: with `on`, the scenery draws into the sun shadow map with its own cull mode (as HD
   draws its map's casters: back faces culled for most), so what faces away from the light casts nothing, as a closed
   room's ceiling seen from below. Characters keep their own. `on` 0, the default, draws every caster's both sides.
   DOL_AURORA_SUN_SHADOW_CASTER_CULL=1 or 0 (local test) overrides it. Takes effect at the next frame. */
void aurora_set_sun_caster_cull(int on);
/* Wind Waker HD's monotone (Frozen Hyrule's sepia, its DOF pass while the game's mMonotone is set). With `on`, at the
   game's grey composite (its I8 copy of the frame to `frame_copy`, mDoGph_gInf_c::mFrameBufferTex, and the quad that
   lays it back, which is left out), the scene's effects run and then HD's pass: the scene's grey (HD's weights) at
   half size, `saturate` of the way, times `mul`, over the scene by clamp(level x clamp((distance - start) / (end -
   start))), all of it beyond `start` when start == end; what the game draws after the composite (the player's
   lists, Link and his things) stays in colour, and bloom goes over all of it, as in HD. `on` 0, the default, leaves
   the game's own grey. Single-sampled only (aurora_monotone_hd_supported()); DOL_AURORA_MONOTONE_HD=0 turns it off.
   Takes effect at the next frame the game records. */
void aurora_set_monotone_hd(int on, float level, float start, float end, const float mul[3], float saturate,
                            uint32_t frame_copy);
int aurora_monotone_hd_supported(void);
/* Wind Waker HD's fog curve: with `on`, the game's linear fog is squared (clamp((d - start) / (end - start))^2, as
   HD's shaders fog), for while the game's palettes hold HD's fog distances, which HD retuned for that curve (on the
   GameCube's linear one they give about twice HD's fog at mid range). `on` 0, the default, keeps the GameCube's
   curve. Takes effect at the next draw. */
void aurora_set_fog_hd(int on);
/* Wind Waker HD's fog starts before the camera in many places (Ganon's Tower -5000, the Tower of the Gods -1000), and
   the GameCube's palette blend (float_kankyo_color_ratio_set) clamps every blended value at 0, so the game's fog
   starts at 0 there and fogs far less than HD. The host gives the stage's HD palette entries with a start below 0
   (count pairs of start, end; at most 8; 0 or NULL: none): where the lit scenery's fog starts at 0 and ends at one
   of those ends, the sun pass fogs it from that start instead, as HD does; every other draw fogged from 0 to the
   first pair's end (characters, plants, objects) takes that start in its own fog. */
void aurora_set_fog_hd_starts(const float* start_end, int count);
/* Wind Waker HD's own normals for the stage in play (the HD pack's optional/hd-normals/<stage>.bin, made by the
   importer from the player's HD dump), for the sun shadows' facing term; NULL or a missing file: none, and
   the facing term makes its own. Loaded at once; call it as the stage changes. */
void aurora_set_hd_normals(const char* path);
/* Wind Waker HD's own vertex colours for the stage in play's rooms (the HD pack's optional/hd-colours/<stage>.bin,
   made by the importer): HD remade its rooms' colours, so the draws with HD's light take HD's (in the game's gamma)
   where a position has one, before the objects' table. A version 2 table also has HD's lava glow (its lava rooms'
   konst2 animation and each vertex's glow), which the lit room draws add outside the shade. NULL or a missing file:
   none. Call it as the stage changes. */
void aurora_set_hd_stage_colours(const char* path);
/* Wind Waker HD's normals and vertex colours for the game's object models (the HD pack's
   optional/hd-normals/objects.bin and optional/hd-colours/objects.bin, made by the importer from the player's HD
   dump): the normals are looked up after the stage's, the colours light draws as HD lights its palms. NULL or a
   missing file: none. Loaded at once; call it once. */
void aurora_set_hd_object_tables(const char* normals_path, const char* colours_path);
/* Wind Waker HD's cloud shadow texture for the cloud shadows on HD's rule (the HD pack's
   optional/hd-cloud-shadow/ww_shadow_clouds.dds, made by the importer from the player's HD dump); NULL or a
   missing file: none, and BlueWake's own noise. Loaded by the render thread at its next frame. */
void aurora_set_hd_cloud_texture(const char* path);
/* Edge light: a warm rim on the sunlit outline of what is near the camera,
   from the same sun and camera. Takes effect at the next frame. */
void aurora_set_edge_light(bool enabled);
/* Torch and fire light: the point lights last passed to aurora_set_point_lights
   (up to 32, each x, y, z, radius, then r, g, b from 0 to 1 and a flicker
   amount, in world space) glow on the surfaces around them, with the camera
   last passed to aurora_set_world_view. Takes effect at the next frame. */
void aurora_set_torch_light(bool enabled);
void aurora_set_point_lights(const float* lights, uint32_t count);
/* Cloud shadows on the sea and the land, from the camera last passed to
   aurora_set_world_view. Takes effect at the next frame. */
void aurora_set_cloud_shadows(bool enabled);
/* The game's camera, for effects that need each pixel's place in the world:
   inv_view, the inverse view matrix (view to world) as three rows of four;
   tan_half_fovy, the tangent of half the vertical field of view; aspect;
   near_plane, the projection's near distance; sea_level, the height of the
   world's sea; seconds, the game's time. */
void aurora_set_world_view(const float inv_view[12], float tan_half_fovy, float aspect, float near_plane,
                           float sea_level, float seconds);
/* The same camera's far distance, for reading distances from the scene's
   depth exactly (0, the default, takes the far plane as infinitely far: at a
   tenth of the far distance that read them a tenth too far, and distant
   surfaces shaded themselves in the sun shadow map). */
void aurora_set_world_view_far(float far_plane);
/* A frames-a-second counter at the top of the window (with the game's own
   rate beside it when in-between frames are on). DOL_AURORA_SHOW_FPS=1 sets
   the starting value. */
void aurora_set_fps_overlay(bool enabled);
/* Where the counter is drawn: the top center (unless set) or a corner. Takes
   effect at the next frame. DOL_AURORA_FPS_POSITION (top-center, top-left,
   top-right, bottom-left or bottom-right) sets the starting value. */
void aurora_set_fps_overlay_position(AuroraFpsOverlayPosition position);
bool aurora_get_frame_interpolation(void);
/* In-between frames per game frame: 1 (60 FPS from the game's 30) or 3
   (120 FPS, for a 120 Hz display). DOL_AURORA_FRAME_INTERP_STEPS sets the
   starting value; a change takes effect at the next game frame. */
void aurora_set_frame_interp_steps(int steps);
int aurora_get_frame_interp_steps(void);
/* Frames presented to the window so far, in-between frames included (what
   the frames-a-second counter counts). */
unsigned long long aurora_get_shown_frames(void);
/* The in-between frames' running totals: game frames seen and interpolated,
   and their draws, the ones rejected as implausible and those with no
   counterpart in the frame before. Differences between two reads cover the
   interval. */
typedef struct AuroraFrameInterpTotals {
  unsigned long long frames, interpolated, draws, rejected, unmatched;
} AuroraFrameInterpTotals;
void aurora_get_frame_interp_totals(AuroraFrameInterpTotals* out);

AuroraBackend aurora_get_backend();
const AuroraBackend* aurora_get_available_backends(size_t* count);

#ifdef __cplusplus
}
#endif

#endif
