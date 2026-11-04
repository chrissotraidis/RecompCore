#include "post_fx.hpp"

#include "dds_io.hpp"

#include <aurora/aurora.h>

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../webgpu/gpu_prof.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace aurora::gfx::post_fx {
using webgpu::g_device;

namespace {
std::atomic_bool g_bloom{false};
std::atomic_bool g_farBlur{false};
std::atomic_bool g_fogHd{false};  // aurora_set_fog_hd
// The lit scenery's fog this frame (note_scene_fog()), for the sun pass; stale counts the post passes since a draw
// gave it (off after a few: a frame without such draws keeps its fog as drawn).
struct SceneFog {
  bool on = false;
  int stale = 0;
  float a = 0.f, b = 0.f, shift = 0.f, c = 0.f;
  float colour[3]{};
  bool square = false;
  bool range = false;           // with GX's range adjust: its table (10 values), the centre and the width
  float table[12]{};
  float centre = 0.f, width = 0.f;
};
std::mutex g_sceneFogMutex;
std::array<float, 16> g_fogHdStarts{};  // aurora_set_fog_hd_starts: start, end pairs (end 0: none)
SceneFog g_sceneFog;
bool g_sunFogHd = true;  // DOL_AURORA_SUN_FOG_HD=0: the scenery's fog as the draws left it, shaded with the rest
std::atomic_bool g_ao{false};
std::atomic_bool g_clouds{false};
std::atomic_bool g_sun{false};
std::atomic_bool g_edge{false};
std::atomic_bool g_torch{false};
float g_torchStrength = 0.35f;  // how bright a light's glow is
float g_torchRadiusScale = 3.f; // the glow's reach, times the radius the game lights with
// The game's point lights (aurora_set_point_lights): x, y, z, radius, r, g, b, flicker each.
std::array<float, kMaxLights * 8> g_lights{};
uint32_t g_lightCount = 0;
float g_edgeStrength = 0.4f;  // how bright a sunlit edge gets
float g_sunStrength = 0.35f;  // how dark a sun shadow is
// The scene's brightness the game asks for (aurora_set_exposure), after the
// HD look's effects and before its bloom: Wind Waker HD's own factor for the
// hour's palettes (its sunlight 3 times at noon on actors and scenery, 1.65
// after gamma). 0 while unset, and the pass is left out. Under g_cameraMutex.
float g_exposureGain = 0.f;
// The frame buffer copied before the exposure, which reads its colours for
// the highlights' roll-off (it can't read the target it draws into).
wgpu::Texture g_exposureScene;
wgpu::TextureView g_exposureSceneView;
// The sea mask (sea_mask_target()) and its own depth.
wgpu::Texture g_seaMask;
wgpu::TextureView g_seaMaskView;
wgpu::Texture g_seaMaskDepth;
wgpu::TextureView g_seaMaskDepthView;
wgpu::Extent3D g_seaMaskSize{};
// The effects' mask (effect_mask_target()), whether it was drawn for the next
// encode(), and a black texel in its place when not.
wgpu::Texture g_effectMask;
wgpu::TextureView g_effectMaskView;
bool g_effectMaskDrawn = false;
wgpu::Texture g_effectNone;
wgpu::TextureView g_effectNoneView;
// DOL_AURORA_EFFECT_MASK_DEBUG=1: the effects the sun shadows leave alone, in
// red (multiplied in where the sun shadows would be, local debug).
bool g_effectDebug = false;
// How much of a pixel's shade (the sun shadows, the occlusion and the cloud
// shadows) its effects take away: their brightest channel times this
// (DOL_AURORA_EFFECT_MASK_GAIN), 2, so where an effect adds half or more (most
// of a glow over the ground) the pixel is left as it is, and under a faint one
// it is mostly shaded with what is behind.
float g_effectGain = 2.f;
// How far back the effects are drawn into the mask (effect_mask_push()).
float g_effectPush = 0.05f;
// The scenery's smooth normals for the facing term (sun_normal_target()).
wgpu::Texture g_sunNormal;
wgpu::TextureView g_sunNormalView;
wgpu::Extent3D g_sunNormalSize{};
// The sun shadow map's full shadow as a colour the scene is multiplied by, the
// shade lit by the sky as in Wind Waker HD: Link's shadow on Windfall's dock at
// noon kept 0.21, 0.39 and 0.56 of the sunlit planks' red, green and blue (0.65
// of each before). DOL_AURORA_SUN_SHADOWS_COLOUR=R,G,B; =0.65,0.65,0.65 is the
// old grey.
float g_sunShadowColour[3] = {0.22f, 0.41f, 0.60f};
// Wind Waker HD's light for the hour (aurora_set_sun_palette, linear): with it, a shaded pixel is multiplied by
// HD's ambient over its sunlight instead of the fixed colour above (which is that ratio at noon, measured by eye).
// Under g_cameraMutex.
struct SunPalette {
  bool on = false;
  float sky[3]{}, ground[3]{}, lit[3]{};
};
SunPalette g_sunPalette;
// Wind Waker HD's light rule from the host (aurora_set_sun_hd_light), under g_cameraMutex.
struct SunHdLight {
  bool on = false;
  bool night = false;
  float fade = 0.f;
};
SunHdLight g_sunHdLight;
// Wind Waker HD's character light's colours for the hour (aurora_set_actor_light_hd, linear), under g_cameraMutex.
struct ActorLightPalette {
  bool on = false;
  float c0[3]{}, shadow[3]{}, k[3]{};
};
ActorLightPalette g_actorLight;
// The GameCube's global actor C0 and K0 (Link's, linear) each character's own toon registers are set against
// (aurora_set_actor_light_hd_own), under g_cameraMutex.
struct ActorLightOwn {
  bool on = false;
  float c0[3]{}, k[3]{};
};
ActorLightOwn g_actorLightOwn;
// The scene pass drew the characters with it this frame (set_actor_light_applied), render thread.
std::atomic_bool g_actorLightApplied{false};
// Post passes encoded so far, and the one that last made the shadow pre-pass (g_sunLow), render thread: the
// characters read it only when the latest one did (actor_shadow_source()).
uint64_t g_postEncodes = 0;
uint64_t g_sunLowEncode = 0;
float g_sunReach = 1500.f;    // how far toward the sun a shadow is looked for, in world units
std::atomic_bool g_sunMapOn{false};
uint32_t g_sunMapSize = 4096;    // the shadow map's texels a side (4096 high, 2048 medium, 1024 low)
float g_sunMapRange = 1250.f;    // world units from the first cascade's centre to its edge (a texel about Link's leg)
float g_sunMapSpread = 4.f;      // each next cascade this many times wider
uint32_t g_sunMapCascades = 3;   // DOL_AURORA_SUN_SHADOW_MAP_CASCADES, 1 to kSunMapCascades
float g_sunMapDepth = 20000.f;   // world units toward and away from the sun it holds, from its centre (at least twice a cascade's range)
float g_sunMapBias = 12.f;       // world units a surface is moved toward the sun before the lookup (first cascade; scaled with the texels)
float g_sunMapContact = 300.f;   // the screen's steps' reach, for what the map leaves out
int g_sunMapDebug = 0;           // DOL_AURORA_SUN_SHADOW_MAP_DEBUG=1: the map's shadow alone, gray outside it; 2: the map itself, depth as contour bands
bool g_sunMapSteps = false;
float g_sunMapSeaLow = 40.f;      // DOL_AURORA_SUN_SHADOW_MAP_SEA_LOW: below the sea's height plus this, a surface counts as sea
float g_sunMapSeaBias = 0.f;      // DOL_AURORA_SUN_SHADOW_MAP_SEA_BIAS: the share of the slope and normal bias the sea keeps (1 before it left the map)
float g_sunMapSoft = 0.01f;       // DOL_AURORA_SUN_SHADOW_MAP_SOFT: the filter's reach grows this many units per unit from the caster
float g_sunMapSoftMax = 4.f;      // DOL_AURORA_SUN_SHADOW_MAP_SOFT_MAX: by at most this many times its own reach again
float g_sunMapSeaRise = 0.f;      // DOL_AURORA_SUN_SHADOW_MAP_SEA_RISE: what stands no higher than this above the sea doesn't shade it (15 before the sea left the map)
bool g_sunMapSmooth = true;
bool g_sunMapGapFill = false;     // DOL_AURORA_SUN_SHADOW_MAP_GAP_FILL=1: steps over the bias's reach (off: they found
                                  // Link's own front through the depth buffer and shaded most of him, which
                                  // Wind Waker HD doesn't; his shadow on the ground keeps its feet without them)
uint32_t g_sunMapTaps = 12;       // DOL_AURORA_SUN_SHADOW_MAP_TAPS: lookups in each shadow's filter disc (6 to 12)
// The shadow quality set while running (set_sun_shadow_quality), 0 low to 2
// high, until the render thread applies it; -1 when nothing is waiting.
std::atomic_int g_sunQualityPending{-1};
float g_sunMapFarStart = 1500.f;  // DOL_AURORA_SUN_SHADOW_MAP_FAR_START/_END: world units from the camera over which the
float g_sunMapFarEnd = 5000.f;    // bias goes from its near values to its far ones
float g_sunMapNormalNear = 2.f;    // DOL_AURORA_SUN_SHADOW_MAP_NORMAL_NEAR: the normal offset near the camera, in texels
                                   // (with 0.5 and 1 below, a wall the sun only grazed was striped light and dark)
float g_sunMapSlopeNear = 20.f;   // DOL_AURORA_SUN_SHADOW_MAP_SLOPE_NEAR: the same near the camera (as far off: at 10,
                                  // Link's side turned from the sun shaded itself from his front, which Wind
                                  // Waker HD's Link doesn't; the sea's own rule keeps shadows at their feet)
float g_sunMapFadeLow = -0.139f;  // DOL_AURORA_SUN_SHADOW_MAP_FADE_LOW/_HIGH: the sun's height (sin of its elevation)
float g_sunMapFadeHigh = -0.122f; // where its shadows are gone, and where they start to go (-8 and -7 degrees). The
                                  // game draws its large sun disk down to -7.5 (06:30-19:30), and HD casts while it shows
float g_sunMapHold = 0.407f;      // DOL_AURORA_SUN_SHADOW_MAP_HOLD: below this height (24 degrees) the shadows are cast
                                  // from the sun held there, as in HD: at the Private Oasis its shadows stopped
                                  // moving between 25 and 23 degrees (17:15-17:25) and stayed put until they went.
                                  // Held at 2 degrees, a low cliff's shadow reached the horizon at sunset
constexpr float kMoonHold = 0.035f; // the moon's, 2 degrees (its timing at dawn was matched with it)
// How dark the sun shadow map's shadows are, as the player chose (1 is HD's;
// aurora_set_sun_shadow_strength): the full shadow's colour is blended in by
// this much, the moon's too.
std::atomic<float> g_sunMapStrength{1.f};
float g_sunMapDim = 0.5f;         // DOL_AURORA_SUN_SHADOW_MAP_DIM: how strong the shadows still are as the sun reaches
                                  // FADE_LOW, from full where they are held: in HD they thin out steadily from there,
                                  // to about half at -7 degrees (a beached boat's shadow on the sand, 18:00-19:30)
// Wind Waker HD's facing term (DOL_AURORA_SUN_SHADOW_FACING=1; off by default while it is tested): what
// faces away from the sun is in shadow as well as what the map shades, as HD's shadow pre-pass makes it from
// normals rebuilt from depth (shadow_pre_pass.baglshpp: is_useDepth2Normal, normal2ShadowMul 10), so a wall
// edge-on to the sun shades smoothly in every cascade instead of by the map's bias at a cascade's edge.
// Characters (the sea mask's green) and the sea keep their own shading. Its distances are HD's pre-pass
// far fade (12000 to 14000).
bool g_sunFacing = false;
float g_sunFacingMul = 10.f;         // DOL_AURORA_SUN_SHADOW_FACING_MUL: fully lit from N.L = 1 / mul (HD: 10)
// With the scenery's smooth normals: the shape's own facing as well, from the depth's normal, as HD's shadow
// pre-pass folds into its shadow term (normal2ShadowMul 10) on top of its models' per-vertex light; 0 off
// (DOL_AURORA_SUN_SHADOW_FACING_DEPTH_MUL, local test).
float g_sunFacingDepthMul = 0.f;
float g_sunFacingDepthBlur = 8.f; // DOL_AURORA_SUN_SHADOW_FACING_DEPTH_BLUR: its blur, pixels at 1080 lines (0 none)
// Scenery takes its ambient's sky-to-ground mix from its smooth normal, as HD's per-vertex light does, not from the
// depth's normal (a face, or a palm's cut-out leaf card) (DOL_AURORA_SUN_SHADOW_SMOOTH_AMBIENT=1, local test;
// =skinned: skinned scenery only, the rest by the depth's normal, as before Nov 3, for comparison).
bool g_sunSmoothAmbient = false;
bool g_sunSmoothAmbientSkinned = false;
// HD's pre-pass takes the farthest of the four depths under each of its pixels (agl's shadow_pre_pass.sh, USE_FETCH_4):
// on a palm's lacy crown that is often what lies behind a leaf, so its leaves barely shade each other
// (DOL_AURORA_SUN_SHADOW_PREPASS_FAR=1, local test; HD's pass only).
bool g_sunPrepassFar = false;
float g_sunFacingFarStart = 12000.f; // DOL_AURORA_SUN_SHADOW_FACING_FAR_START / _END: where it fades out
float g_sunFacingFarEnd = 14000.f;
// Where the facing term fades out (end: its end, else its start). With the cascades following the camera (g_follow,
// HD's pass) it doesn't, unless DOL_AURORA_SUN_SHADOW_FACING_FAR_START is set: it is HD's light amount (its per-vertex
// N.L, R2.w), which HD doesn't fade with distance; only HD's pre-pass shadow fades out at 12000 to 14000. Faded, the
// title's island, beyond 14000 for its first seconds, stood flat and lit and grew its shade as the camera came in.
float facing_far(bool end) noexcept;
// With the facing term, HD's terrain curve for its shadow (its terrain shaders' 0.5 x pow(2 x shadow, e), e
// 4 near and 2 farther: full light from half shadow up, almost none below a quarter); the distance of that
// change isn't read yet (its uniform was out of the probe's range), so 300 to 500 is a guess.
bool g_sunShape = true;              // DOL_AURORA_SUN_SHADOW_SHAPE=0 leaves the map's own soft edge
float g_sunShapeNear = 300.f;        // DOL_AURORA_SUN_SHADOW_SHAPE_NEAR / _FAR
float g_sunShapeFar = 500.f;
float g_moonStrength = 0.5f;      // DOL_AURORA_MOON_SHADOWS_STRENGTH: the moon's shadows, a share of the sun's
float g_moonFadeLow = 0.42f;      // DOL_AURORA_MOON_SHADOWS_FADE_LOW/_HIGH: the same for the moon (25 and 30
float g_moonFadeHigh = 0.5f;      // degrees) as the night comes on, as in HD
float g_moonDawnLow = 0.139f;     // DOL_AURORA_MOON_SHADOWS_DAWN_LOW/_HIGH: the same at dawn, when HD keeps them down
float g_moonDawnHigh = 0.17f;     // to 8 degrees (the moon setting in the west), then hands over to the sun
// GLSL's smoothstep.
float smoothstep_f(float lo, float hi, float x) noexcept {
  const float t = std::clamp((x - lo) / (hi - lo), 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}
// DOL_AURORA_SUN_SHADOW_MAP_HD_BIAS=1 (local test): Wind Waker HD's bias in place of the slope and normal terms
// (the lookup moved out 20 x (1 - |N.L|)^9; the casters' depth biased when drawn, gxcore_draw.cpp).
bool g_sunMapHdBias = false;
// DOL_AURORA_SUN_SHADOW_HD_MIX=1 (local test): Wind Waker HD's terrain shadow mix (the shadow at a quarter of
// the size, blurred as HD's pre-pass is, taken by distance and grazing, and HD's curve). Its fades,
// DOL_AURORA_SUN_SHADOW_HD_MIX_FADES=curve,blur start,blur end (HD's at Outset's bridge at 16:00: 545,1318,1833),
// and DOL_AURORA_SUN_SHADOW_HD_MIX_BLURS (blur passes, X then Y each; 1).
bool g_sunHdMix = false;
float g_sunHdMixFades[3] = {545.f, 1318.f, 1833.f};
int g_sunHdMixBlurs = 1;
// DOL_AURORA_SUN_SHADOW_HD_MIX_NEAR_POWER: the curve's power near the camera (HD's 4). With BlueWake's filter it
// sharpens the dithered soft edge of Link's shadow into specks; 2 or 1 keeps the edge soft (local test).
float g_sunHdMixNearPower = 4.f;
// The "HD" shadow quality (DOL_AURORA_SUN_SHADOW_QUALITY=hd, set_sun_shadow_quality(3)): Wind Waker HD's shadow pass
// as a RenderDoc capture of Cemu shows it (Outset's bridge, 10:00): a 1024 x 1024 16-bit map for each of 3
// cascades, the first fitted to 1222 units across (here a square that wide), the others reaching 1395 and 14000
// units ahead (HD's are perspective-warped; here squares that reach as far); a pre-pass at half the screen's size,
// one hardware-filtered lookup per pixel in the cascade its distance picks, blurred, which every pixel reads; with
// HD's bias and terrain mix.
bool g_sunHdPrepass = false;
// DOL_AURORA_SUN_SHADOW_SEA_STRENGTH: the sea's shadow strength (HD's cPrimShadowIntensity, read in the capture: 0.5,
// HD's pass's default; 1 otherwise), where the sea mask has the sea. -1: the default.
float g_sunSeaShadow = -1.f;
float g_sunMapRanges[kSunMapCascades] = {}; // each cascade's range when set (0: g_sunMapRange x g_sunMapSpread^k)
// HD's pass: the distances from the camera where its pre-pass moves to the second and third cascade (HD's 420 and
// 1395; DOL_AURORA_SUN_SHADOW_HD_SPLITS=a,b, local test, with _MAP_RANGES for a camera zoomed out past HD's).
float g_sunHdSplits[2] = {420.f, 1395.f};
// HD's splits this frame, from the game's camera (aurora_set_sun_cascade_splits; 0: g_sunHdSplits). HD sets them each
// frame from the camera's distance to its target and its field of view, and fits its first two cascades to the
// slices they make: with fixed ones, BlueWake's camera zoomed out (the mouse wheel's 2x) left the island in the far
// cascade's 18-unit texels and its grass's shadows crawled as the sun turned (Star Island, Nov 1).
std::atomic<float> g_sunHdLiveSplits[2]{0.f, 0.f};
// Cascades that follow the camera (DOL_AURORA_SUN_SHADOW_CASCADES_FOLLOW; on by default with HD's pass, =0 for the
// cascades fixed to the sun and fitted around the camera's place). As Wind Waker HD's (two Windfall captures: its
// shadow casters' x axis is the camera's right turned flat to the sun, its first cascade a box fitted to the slice
// its first split makes): each cascade is fitted to the slice of the view between two splits, so all that is on
// screen is in one (the title's island far off was outside them all and grew its shade as the camera came near),
// turned with the camera. Unlike HD's, each one's size is held in steps (a quarter of a power of two, rounded up,
// shrinking only two steps down) and its centre moved by whole texels in its own turned frame, so its texels stay
// where they were on the ground while the camera moves and doesn't turn: resized every frame with the live splits
// (since 9031051), every shadow near the camera was drawn on a new grid each frame and small ones flashed.
// _FAR: where the last ends (HD's 14000, the camera's far plane if nearer), its last seventh fading to lit as HD's
// pre-pass far fade does (12000 to 14000). _SPLIT3: where a fourth cascade starts (0: three; a third reaching
// 14000 across the whole view would have texels of 30 units): at least 1.5 times HD's far split.
// _TURN_STEP: degrees the frame turns in (0: with every turn of the camera, as HD's).
bool g_follow = false;
float g_followFar = 14000.f;
float g_followSplit3 = 4500.f;
float g_followTurnStep = 0.f;
struct FollowCascades {
  float held[kSunMapCascades][2]{}; // each cascade's half sizes across (x, y), held between steps
  float turn = 0.f;                 // the frame's turn about the sun (radians) as held
  bool turnHeld = false;
  // This frame's, for the uniform: each cascade's larger half size and its depth's reach.
  float range[kSunMapCascades]{};
  float depth[kSunMapCascades]{};
  float split3 = 0.f, fadeStart = 0.f, fadeEnd = 0.f;
  bool on = false; // this frame's map was fitted so
};
FollowCascades g_followState;
float facing_far(bool end) noexcept {
  if (g_follow && std::getenv("DOL_AURORA_SUN_SHADOW_FACING_FAR_START") == nullptr)
    return end ? 1e7f : 1e7f - 1.f;
  return end ? g_sunFacingFarEnd : g_sunFacingFarStart;
}

// HD's split k (0: its second cascade's start, 1: its third's) this frame.
float hd_split(int k) noexcept {
  const float live = g_sunHdLiveSplits[k].load(std::memory_order_relaxed);
  return live > 0.f ? live : g_sunHdSplits[k];
}
// DOL_AURORA_SUN_SHADOW_TEMPORAL=1 (local test; with HD's terrain mix or pass): the quarter-size pre-pass blended
// with the last frames' at the same place in the world, more the farther from the camera
// (DOL_AURORA_SUN_SHADOW_TEMPORAL_RANGE=near,far,most: none nearer than `near`, `most` of the last frames' from `far`
// on). At sea the game's clock turns the sun a little every frame, and each texel of the 1024 map an edge crosses
// showed as a jump (Sector 0A at 14:00: the tree's, the sign's and the bridge rail's shadows wiggled with the
// camera still, and stood still with the clock held). Last frame's place comes from its camera and the depth the
// blend keeps in alpha; a place that was hidden then (another depth there) takes this frame's alone.
bool g_sunTemporal = false;
bool g_sunTemporalHeld = false; // encode_late()'s lighting: no blend with the last frame, no new history
// DOL_AURORA_SUN_SHADOW_SKINNED_SCENERY=1 (local test): only the toon-shaded draws (the mask's green 1: Link, the
// islanders) are characters to the sun pass; the skinned-only (green 0.6: a palm's swaying leaves) are scenery.
// Wind Waker HD lights Outset's palms with its terrain shader (a RenderDoc capture at the bridge: the scenery's
// light x 3, per-vertex N.L, the shadow pre-pass and the cloud texture); as characters here their leaves took no
// facing term or terrain mix and came out flat green.
bool g_sunSkinnedScenery = false;
float g_sunTemporalRange[3] = {300.f, 1500.f, 0.9f};
wgpu::Texture g_sunHist;
wgpu::TextureView g_sunHistView;
bool g_sunHistValid = false;
std::array<float, 16> g_sunHistCamera{}; // last frame's Camera::view (its view to world rows, its lens)
float g_sunHistTime = 0.f;               // and its seconds
wgpu::ShaderModule g_sunTemporalModule;
wgpu::BindGroupLayout g_sunTemporalLayout;
wgpu::PipelineLayout g_sunTemporalPipelineLayout;
wgpu::RenderPipeline g_sunTemporalPipeline;
// HD's terrain mix is on: asked for (DOL_AURORA_SUN_SHADOW_HD_MIX) or part of HD's pass.
bool hd_mix_on() noexcept { return g_sunHdMix || g_sunHdPrepass; }
// DOL_AURORA_AO_HD=1 (local test; needs DOL_AURORA_SUN_SHADOW_HD_MIX=1 and the AO on): Wind Waker HD's SSAO in
// place of BlueWake's, made in the terrain mix's quarter-size pass (blurred with its shadow) and applied, as HD
// does, to the scenery's shade alone: nothing in full sun, nor on characters or the sea. Its values are HD's
// (DOL_AURORA_AO_HD_RADIUS, _FAR, _ATTN, _DENSITY, _OFFSET); _POWER the blurred AO's power (3, for HD's sharp x
// blurred squared), _FACING=0 leaves out HD's fade on faces square to the camera, _DEBUG=1 shows it alone.
bool g_aoHd = false;
float g_aoHdRadius = 0.03f;
float g_aoHdFar = 10000.f;
float g_aoHdAttn = 320.f;
float g_aoHdDensity = 1.1f;
float g_aoHdOffset = 0.001f;
float g_aoHdPower = 3.f;
bool g_aoHdFacing = true;
bool g_aoHdDebug = false;
// DOL_AURORA_CLOUD_SHADOWS_HD=1 (local test; with the sun shadow map and the cloud shadows on): the cloud shadows
// on Wind Waker HD's rule, in the sun shadows' pass in place of their own: they take away the sun's light (HD's
// shadow term x clamp(cloud + floor)), so they don't darken shade, and are lighter at a low sun.
// DOL_AURORA_CLOUD_SHADOWS_HD_POWER sets the floor (HD's cCloudShadowPower, else HD's from the game's clock as the
// host gives it, or 1 - max(0.25, 0.9 x the sun's height): cloud_hd_floor()).
bool g_cloudHd = false;
float g_cloudHdPower = -1.f;
// The floor from the host's clock (aurora_set_cloud_hd_floor), negative while unset.
std::atomic<float> g_cloudHdFloor{-1.f};
// DOL_AURORA_CLOUD_SHADOWS_HD_TEXTURE=path to a BC4 DDS of Wind Waker HD's ww_shadow_clouds (the sea stage's
// Stage.bfres, 4096 x 4096): with it, the clouds are HD's as a RenderDoc capture and Cemu's memory show them, the
// texture seen from above, a repeat every 15,000 units (cCloudProjMtx: u = -x / 15000 + u0, v = -z / 15000 + 0.5),
// u0 growing 0.01116 a second (Cemu, Outset at 10:00, 40 s); else BlueWake's noise. _SPEED and _REPEAT for tests.
wgpu::Texture g_cloudTex;
wgpu::TextureView g_cloudTexView;
wgpu::Sampler g_cloudSampler; // repeating, trilinear
std::string g_cloudTexPath;   // aurora_set_hd_cloud_texture's, under g_cameraMutex
bool g_cloudTexPending = false; // given and not loaded yet
void load_cloud_texture(const char* path);
float g_cloudHdSpeed = 0.01116f;
float g_cloudHdRepeat = 15000.f;
constexpr wgpu::TextureFormat kSunLowFormat = wgpu::TextureFormat::RGBA16Float;
wgpu::Texture g_sunLow[2];
wgpu::TextureView g_sunLowView[2];       // both mips, to read
wgpu::TextureView g_sunLowMipView[2][2]; // [texture][mip], to draw into (mip 1: HD's pass's mip 1)
wgpu::Extent3D g_sunLowSize{};
wgpu::RenderPipeline g_sunMapLow;
wgpu::ShaderModule g_sunBlurModule;
wgpu::BindGroupLayout g_sunBlurLayout;
wgpu::PipelineLayout g_sunBlurPipelineLayout;
wgpu::RenderPipeline g_sunBlurX, g_sunBlurY;
wgpu::RenderPipeline g_sunCopy; // one bilinear tap (make_sun_blur_pipelines)
// The characters' shadow from this frame (draw_actor_shadow()): HD's pre-pass at half the scene's size (mip 0) and
// its mip 1, a blurred copy at an eighth brought back up; a half-size scratch for the blur, two eighth-size ones.
// g_actorLowReady: made for the pass being encoded (actor_shadow_source()).
wgpu::Texture g_actorLow;
wgpu::TextureView g_actorLowView;     // both mips, to read
wgpu::TextureView g_actorLowMip[2];   // one mip each, to draw into
wgpu::Texture g_actorTmp[3];          // half, eighth, eighth
wgpu::TextureView g_actorTmpView[3];
wgpu::Extent3D g_actorLowSize{};
bool g_actorLowReady = false;
float g_sunMapSlope = 20.f;       // DOL_AURORA_SUN_SHADOW_MAP_SLOPE: texels of the surface's own depth slope added to the bias
                                  // far off (10 left acne in stripes on Ice Ring Isle's walls edge-on to the sun)
float g_sunMapNormalOffset = 2.f; // DOL_AURORA_SUN_SHADOW_MAP_NORMAL_OFFSET: a lookup moved out along the surface's normal, in its cascade's texels      // DOL_AURORA_SUN_SHADOW_MAP_STEPS=1: the screen's steps too, for what the map leaves out
// A cascade's range (world units from its centre to its edge), and the depth
// toward and away from the sun it holds: enough for what stands in it.
float cascade_range(uint32_t cascade) noexcept {
  if (g_followState.on && cascade < kSunMapCascades && g_followState.range[cascade] > 0.f)
    return g_followState.range[cascade];
  if (cascade < kSunMapCascades && g_sunMapRanges[cascade] > 0.f) {
    // HD's first two are fitted to the slices its splits make this frame (the ranges are for its splits at Outset's
    // bridge, g_sunHdSplits); its far one already reaches past the haze.
    const float scale = g_sunHdPrepass && cascade < 2 ? hd_split(static_cast<int>(cascade)) / g_sunHdSplits[cascade] : 1.f;
    return g_sunMapRanges[cascade] * scale;
  }
  return g_sunMapRange * std::pow(g_sunMapSpread, static_cast<float>(cascade));
}
float cascade_depth(float range) noexcept { return std::max(g_sunMapDepth, 2.f * range); }
// Cascade k's depth's reach toward and away from the sun from its centre.
float cascade_depth_of(uint32_t cascade) noexcept {
  if (g_followState.on && cascade < kSunMapCascades && g_followState.depth[cascade] > 0.f)
    return g_followState.depth[cascade];
  return cascade_depth(cascade_range(cascade));
}
// The sun shadow pass's uniform (kSunMapShader's View).
// 100 floats of the view, then Wind Waker HD's light buffer for the scenery's shade (points, pl: 4 + kMaxLights x 8).
// Then the lit scenery's fog (fog, fog_colour, fog_x, fog_range: 24), and HD's fog starts below 0 (fog_starts: 16).
constexpr size_t kSunMapUniformFloats = 104 + kMaxLights * 8 + 24 + 16;
float g_cloudStrength = 0.28f;  // how dark a cloud's shadow is
float g_cloudScale = 7000.f;    // world units across one cloud
bool g_cloudDebug = false;
// The camera, for the cloud shadows (aurora_set_world_view): the inverse view
// matrix's three rows, then tan(fovy / 2), aspect, near, the sea's height, and
// the time in seconds, then the direction to the sun (aurora_set_sun_direction).
// Written by the game thread, read when a frame is encoded.
std::mutex g_cameraMutex;
std::array<float, 20> g_camera{};
float g_cameraFar = 0.f;  // aurora_set_world_view_far; 0: infinitely far
bool g_cameraValid = false;
float g_aoRadius = 30.f;    // the occlusion radius, in near-plane units
float g_aoStrength = 0.6f;  // how dark a fully occluded spot gets
float g_aoFadeStart = 1000.f;  // distance in near-plane units where the occlusion starts to fade
float g_aoFadeEnd = 2500.f;    // and where it is gone (the sea's swells beyond darkened it in a band)
int g_aoDebug = 0;  // DOL_AURORA_AO_DEBUG=1: the occlusion alone; 2: what it darkens tinted red over the scene
float g_farStart = 8000.f;  // distance in near-plane units where the far blur starts
float g_farEnd = 30000.f;   // and where it is full
float g_farAmount = 0.8f;  // how much of the blurred scene at full
float g_farBackground = 0.35f;  // and where there is no depth (the sky, the horizon)
// DOL_AURORA_FAR_BLUR_HD=1: Wind Waker HD's ramp (wwHD_basic's DOF) in place of the smoothstep: the blend is
// clamp(level x (distance - start) / (end - start)), times the amount, with the level the host gives for the hour
// (aurora_set_far_blur_level: HD's 1.2 by day, 0.65 at night).
bool g_farHd = false;
std::atomic<float> g_farLevel{1.2f};
bool g_farDebug = false;
float g_threshold = 0.72f;
float g_strength = 0.35f;
// DOL_AURORA_BLOOM_HD=1 (local test): Wind Waker HD's bloom chain (agl's bloom.sh, its values read from a
// RenderDoc capture of Cemu) in place of the blur above: what is bright at a quarter of the size x 0.65, then
// three levels at 1/8, 1/16 and 1/32 of the size, each blurred by HD's 13-tap Gaussian, tinted warm, lavender
// and blue, gathered and screened over the scene. DOL_AURORA_BLOOM_HD_MASK=1 adds HD's mask, a first cut: only
// the sky glows (what no solid draw covers, or what writes no depth), from the sea mask's pass (its blue, 0.5 there
// and 0 elsewhere: HD's bloom offsets 0 and -1). HD's other glowing materials (lit windows, fire) need its table.
bool g_bloomHd = false;
bool g_bloomHdMask = false;
float g_bloomHdIntensity = 0.65f; // _INTENSITY: HD's cThresholdParam.z
float g_bloomHdThreshold = -1.f;  // _THRESHOLD and _RANGE: the brightness the glow starts at and over which it is
float g_bloomHdRange = -1.f;      // full (HD's 0 and 0.1, with the mask; without it, BlueWake's 0.72 and 0.28)
float g_bloomHdScale = 2.f;       // _SCALE: HD's cLuminanceParam.x (the mask's offset times this)
float g_bloomHdStrength = 1.f;    // _STRENGTH: the glow times this before it is screened
// _COLOR1/2/3=r,g,b: each level's tint (color1 on 1/8, color2 on 1/16, color3 on 1/32), from the capture.
float g_bloomHdColors[3][3] = {{1.f, 0.828f, 0.454f}, {0.523f, 0.575f, 0.765f}, {0.f, 0.33f, 1.f}};
int g_bloomHdDebug = 0; // _DEBUG=1: the glow alone; 2: where the mask lets a glow start, red over the scene

wgpu::ShaderModule g_module;
wgpu::BindGroupLayout g_layout;
wgpu::PipelineLayout g_pipelineLayout;
wgpu::Sampler g_sampler;
wgpu::TextureFormat g_format = wgpu::TextureFormat::Undefined;
wgpu::RenderPipeline g_bright, g_blurH, g_blurV;
wgpu::RenderPipeline g_composite;
wgpu::RenderPipeline g_exposureDepth; // the scene brightened (g_exposureGain), where there is depth
wgpu::RenderPipeline g_down;           // the scene at a quarter of the size, unfiltered
wgpu::RenderPipeline g_farComposite;   // blended into the EFB by distance
wgpu::RenderPipeline g_aoPass;         // occlusion at a quarter of the size
wgpu::RenderPipeline g_aoBlurH, g_aoBlurV;  // blurred along each surface
wgpu::RenderPipeline g_aoUpscale;      // multiplied into the EFB along each surface
wgpu::RenderPipeline g_aoApply;        // multiplied into the EFB (sun shadows)
uint32_t g_aoSamples = 0;
std::array<wgpu::Texture, 2> g_aoTex;
std::array<wgpu::TextureView, 2> g_aoView;
uint32_t g_farSamples = 0;
wgpu::BindGroupLayout g_depthLayout;
wgpu::PipelineLayout g_depthPipelineLayout;
// The far blur's composite: g_depthLayout's sampler, blurred scene and depth, then its uniform (HD's level) at 3 and
// an unused texture at 4, as draw_with_uniform() binds them.
wgpu::BindGroupLayout g_farLayout;
wgpu::PipelineLayout g_farPipelineLayout;
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

// Half an 8-bit step of noise (interleaved gradient noise): a smooth shade
// multiplied or added into the 8-bit scene otherwise shows as bands.
fn dither(pos: vec2<f32>) -> f32 {
    return (fract(52.9829189 * fract(dot(pos, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
}

// Multiplied into the scene (sun shadows); the EFB's alpha is kept.
@fragment
fn fs_ao_apply(in: VertexOutput) -> @location(0) vec4<f32> {
    let shade = textureSample(src, samp, in.uv).rgb;
    return vec4(shade + select(0.0, dither(in.pos.xy), shade.g < 0.999), 1.0);
}

// Screened onto the scene (glow * (1 - scene) + scene); the EFB's alpha is kept.
@fragment
fn fs_composite(in: VertexOutput) -> @location(0) vec4<f32> {
    let glow = textureSample(src, samp, in.uv).rgb * strength;
    return vec4(max(glow + select(0.0, dither(in.pos.xy), max(glow.r, max(glow.g, glow.b)) > 0.002), vec3(0.0)), 0.0);
}
)""";

// Wind Waker HD's bloom chain (g_bloomHd). src2: the sea mask for the detect and the compose's debug view, the
// smaller level for a gather.
constexpr const char* kBloomHdShader = R"""(
override intensity: f32 = 0.65;
override threshold: f32 = 0.0;
override range: f32 = 0.1;
override use_mask: f32 = 0.0;
override mask_scale: f32 = 2.0;
override axis_x: f32 = 1.0;
override tint_ar: f32 = 1.0;
override tint_ag: f32 = 1.0;
override tint_ab: f32 = 1.0;
override tint_br: f32 = 1.0;
override tint_bg: f32 = 1.0;
override tint_bb: f32 = 1.0;
override strength: f32 = 1.0;
override debug_view: f32 = 0.0;

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var src2: texture_2d<f32>;

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

// Four bilinear taps a texel of `t` out each way: a 4x4 block of it (HD's BLM_DIRECT_REDUCE).
fn taps4(t: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let o = 1.0 / vec2<f32>(textureDimensions(t));
    return 0.25 * (textureSampleLevel(t, samp, uv + o * vec2(-1.0, -1.0), 0.0) +
                   textureSampleLevel(t, samp, uv + o * vec2(1.0, -1.0), 0.0) +
                   textureSampleLevel(t, samp, uv + o * vec2(-1.0, 1.0), 0.0) +
                   textureSampleLevel(t, samp, uv + o * vec2(1.0, 1.0), 0.0));
}

// HD's detect: a = clamp((luma - threshold) / range), then the mask's offset (0.5 none, 0 -1) x mask_scale.
@fragment
fn fs_detect(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = taps4(src, in.uv).rgb;
    let luma = dot(c, vec3(0.2989, 0.5866, 0.1145));
    var a = clamp((luma - threshold) / range, 0.0, 1.0);
    if (use_mask > 0.5) {
        a = max(a + (taps4(src2, in.uv).b - 0.5) * mask_scale, 0.0);
    }
    return vec4(c * a * intensity, 1.0);
}

@fragment
fn fs_down(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(taps4(src, in.uv).rgb, 1.0);
}

// HD's 13-tap binomial Gaussian (agl_gaussian.shinc, BLUR_KERNEL 13) as 7 bilinear taps, one axis a pass.
@fragment
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    let o = vec2<f32>(axis_x, 1.0 - axis_x) / vec2<f32>(textureDimensions(src));
    var c = 0.2256 * textureSampleLevel(src, samp, in.uv, 0.0).rgb;
    c += 0.3142 * (textureSampleLevel(src, samp, in.uv + o * 1.3846, 0.0).rgb +
                   textureSampleLevel(src, samp, in.uv - o * 1.3846, 0.0).rgb);
    c += 0.0698 * (textureSampleLevel(src, samp, in.uv + o * 3.2308, 0.0).rgb +
                   textureSampleLevel(src, samp, in.uv - o * 3.2308, 0.0).rgb);
    c += 0.0032 * (textureSampleLevel(src, samp, in.uv + o * 5.0769, 0.0).rgb +
                   textureSampleLevel(src, samp, in.uv - o * 5.0769, 0.0).rgb);
    return vec4(c, 1.0);
}

// HD's gather, as captured: this level x its tint plus the smaller one x its own.
@fragment
fn fs_gather(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(textureSampleLevel(src, samp, in.uv, 0.0).rgb * vec3(tint_ar, tint_ag, tint_ab) +
                textureSampleLevel(src2, samp, in.uv, 0.0).rgb * vec3(tint_br, tint_bg, tint_bb), 1.0);
}

fn dither(pos: vec2<f32>) -> f32 {
    return (fract(52.9829189 * fract(dot(pos, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
}

// Screened onto the scene, as HD's compose (src one minus the scene, dst one); the EFB's alpha is kept.
@fragment
fn fs_compose(in: VertexOutput) -> @location(0) vec4<f32> {
    if (debug_view > 1.5) {
        return vec4(textureSampleLevel(src2, samp, in.uv, 0.0).b * 1.6, 0.0, 0.0, 0.0);
    }
    let glow = textureSampleLevel(src, samp, in.uv, 0.0).rgb * strength;
    return vec4(max(glow + select(0.0, dither(in.pos.xy), max(glow.r, max(glow.g, glow.b)) > 0.002), vec3(0.0)), 0.0);
}
)""";
wgpu::ShaderModule g_bloomHdModule;
wgpu::BindGroupLayout g_bloomHdLayout;
wgpu::PipelineLayout g_bloomHdPipelineLayout;
wgpu::TextureFormat g_bloomHdFormat = wgpu::TextureFormat::Undefined;
uint32_t g_bloomHdSamples = 0;
wgpu::RenderPipeline g_bloomHdDetect, g_bloomHdDetectMasked, g_bloomHdDown, g_bloomHdBlurX, g_bloomHdBlurY;
wgpu::RenderPipeline g_bloomHdGather1, g_bloomHdGather2, g_bloomHdCompose;
// The detect at 1/4 of the size, then each level (1/8, 1/16, 1/32) twice, for its blur.
std::array<wgpu::Texture, 7> g_bloomHdTex;
std::array<wgpu::TextureView, 7> g_bloomHdView;
wgpu::Extent3D g_bloomHdSize{};

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
// Wind Waker HD's ramp (g_farHd): clamp(level x (distance - start) / (end - start)) x amount, linear, with HD's DOF
// level for the hour in far_u.x (1.2 by day: full at 45,833 of 25,000 to 50,000; 0.65 at night: 0.65 at 50,000).
override hd_ramp: bool = false;

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var blurred: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> far_u: vec4<f32>;

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
    let dist = 1.0 / max(d, 1e-9);
    var ramp = smoothstep(far_start, far_end, dist);
    if (hd_ramp) {
        ramp = clamp(far_u.x * clamp((dist - far_start) / (far_end - far_start), 0.0, 1.0), 0.0, 1.0);
    }
    var f = select(ramp * amount, background, d <= 0.0);
    // Within the blur's reach of where there is no depth, less of it, as much
    // less as the sky is left sharp: there the blurred picture is part sky,
    // and with the sky sharp the sea along the horizon took a bright band and
    // a distant lightning strike (it has depth) spread into a wide grey column
    // from the sea to the clouds. Wind Waker HD keeps the line sharp.
    if (d > 0.0 && background < amount) {
        let size = vec2<i32>(textureDimensions(depth));
        var sky = 0.0;
        for (var r = 1; r <= 3; r++) {
            let o = r * 8;
            let p = vec2<i32>(in.pos.xy);
            sky += select(0.0, 1.0, textureLoad(depth, clamp(p + vec2(o, 0), vec2(0), size - 1), 0) <= 0.0);
            sky += select(0.0, 1.0, textureLoad(depth, clamp(p - vec2(o, 0), vec2(0), size - 1), 0) <= 0.0);
            sky += select(0.0, 1.0, textureLoad(depth, clamp(p + vec2(0, o), vec2(0), size - 1), 0) <= 0.0);
            sky += select(0.0, 1.0, textureLoad(depth, clamp(p - vec2(0, o), vec2(0), size - 1), 0) <= 0.0);
        }
        f = mix(f, background, min(sky / 4.0, 1.0));
    }
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

// Ambient occlusion from depth alone: around each pixel, twelve samples in a
// ring whose size on screen shrinks with distance; a sample above the pixel's
// surface (from the depth around it) and within the radius of it in the scene
// occludes it. Where there is no depth (the sky, the horizon) nothing is occluded. Written at a quarter of the
// size, blurred, then multiplied into the scene (fs_ao_upscale). The blur and
// the upscale take only what is at about the same distance: blurred and
// stretched as one, Link's own shade spread onto the sea behind him, a dark
// outline around him (#491).
constexpr const char* kAoShader = R"""(
override radius: f32 = 30.0;
override strength: f32 = 0.6;
override fade_start: f32 = 1000.0;
override fade_end: f32 = 2500.0;
override debug_view: f32 = 0.0;
override effect_gain: f32 = 2.0;

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var effects: texture_2d<f32>; // the effects' mask, which the occlusion leaves
var<private> weights: array<f32, 5> = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

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

fn distance_at(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth));
    let p = clamp(vec2<i32>(uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let d = textureLoad(depth, p, 0);
    return select(1.0 / d, 0.0, d <= 0.0);
}

// The distance at one pixel of the depth, 0 where there is none (or off the
// picture).
fn depth_distance(p: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth));
    if (any(p < vec2<i32>(0)) || any(p >= size)) {
        return 0.0;
    }
    let d = textureLoad(depth, p, 0);
    return select(1.0 / d, 0.0, d <= 0.0);
}

@fragment
fn fs_ao(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(depth));
    // Whole pixels of the depth, each placed at its centre: a position taken
    // from where a sample was aimed rather than the pixel it read, and a
    // neighbour found by adding a pixel's width to a position on a pixel's
    // edge (sometimes the same pixel again), bent the surface in rows on a
    // floor and in columns on a wall seen from the side.
    let p = vec2<i32>(floor(in.uv * size));
    let dist = depth_distance(p);
    // Faded out with distance: further away, the ring is a few pixels across
    // and the sea's swells darkened it in a band toward the horizon.
    let fade = 1.0 - smoothstep(fade_start, fade_end, dist);
    if (dist <= 0.0 || fade <= 0.0) {
        return vec4(1.0);
    }
    // About a 60-degree view: a radius r at distance z covers 0.87 r / z of the height.
    let reach = vec2(size.y / size.x, 1.0) * (0.87 * radius / dist);
    let here = texel_position(p, dist, size);
    let normal = surface_normal(p, here, size);
    // Six directions, a sample on each side. A sample shades the pixel when it
    // is above the pixel's surface and within the radius of it in the scene,
    // not just on screen: across a floor seen at a low angle a short step on
    // screen is a long way along the floor, and Link's legs darkened floor well
    // in front of him. A flat surface is not shaded at any angle, and something
    // far in front (a silhouette against the background) is out of reach.
    var occluded = 0.0;
    for (var i = 0; i < 12; i++) {
        let a = f32(i % 6) * 0.5236 + 0.26 + select(0.0, 3.14159, i >= 6);  // 30 degrees apart, both sides
        let r = (f32(i % 3) + 1.0) / 3.0;
        let q = vec2<i32>(floor((in.uv + vec2(cos(a), sin(a)) * reach * r) * size));
        let n = depth_distance(q);
        if (n > 0.0) {
            let v = texel_position(q, n, size) - here;
            let l2 = dot(v, v);
            if (l2 < radius * radius && l2 > 1e-6) {
                // How far above the surface, as the sine of the angle (a little
                // left out for the depth's own steps), less with distance.
                let rise = max(dot(v, normal) * inverseSqrt(l2) - 0.1, 0.0);
                occluded += rise * (1.0 - l2 / (radius * radius));
            }
        }
    }
    let ao = 1.0 - strength * fade * clamp(occluded / 4.0, 0.0, 1.0);
    return vec4(vec3(ao), 1.0);
}

// A pixel's place in view space, in near-plane units, with the same 60-degree
// view as the reach above (a different field of view only stretches the
// picture sideways, so a flat surface stays flat).
fn view_position(uv: vec2<f32>, dist: f32, size: vec2<f32>) -> vec3<f32> {
    let ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * 0.57735 * size.x / size.y, -ndc.y * 0.57735, 1.0) * dist;
}

fn texel_position(p: vec2<i32>, dist: f32, size: vec2<f32>) -> vec3<f32> {
    return view_position((vec2<f32>(p) + 0.5) / size, dist, size);
}

// The surface's normal from the depth around the pixel, facing the camera.
// On each axis the side nearer the pixel's distance, so an edge (Link in front
// of the floor) does not bend it.
fn surface_normal(p: vec2<i32>, here: vec3<f32>, size: vec2<f32>) -> vec3<f32> {
    let dx = side(p, here, vec2<i32>(1, 0), size);
    let dy = side(p, here, vec2<i32>(0, 1), size);
    var n = normalize(cross(dx, dy));
    if (dot(n, here) > 0.0) {
        n = -n;
    }
    return n;
}

fn side(p: vec2<i32>, here: vec3<f32>, delta: vec2<i32>, size: vec2<f32>) -> vec3<f32> {
    let a = depth_distance(p + delta);
    let b = depth_distance(p - delta);
    let da = select(abs(a - here.z), 3.4e38, a <= 0.0);
    let db = select(abs(b - here.z), 3.4e38, b <= 0.0);
    if (da <= db) {
        return texel_position(p + delta, select(a, here.z, a <= 0.0), size) - here;
    }
    return here - texel_position(p - delta, select(b, here.z, b <= 0.0), size);
}

// How much a neighbour's shade counts at a distance: fully on the same
// surface, nothing across an edge to something nearer or further away, or
// where there is no depth.
fn same_surface(dist: f32, other: f32) -> f32 {
    let e = (other - dist) / (0.25 * dist + radius);
    return select(exp(-e * e), 0.0, other <= 0.0);
}

// The quarter-size texel's distance: where fs_ao read it.
fn texel_distance(t: vec2<i32>) -> f32 {
    return distance_at((vec2<f32>(t) + 0.5) / vec2<f32>(textureDimensions(src)));
}

fn blur(p: vec2<i32>, dir: vec2<i32>) -> vec4<f32> {
    let dist = texel_distance(p);
    if (dist <= 0.0) {
        return vec4(1.0);
    }
    let last = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
    var sum = textureLoad(src, p, 0).r * weights[0];
    var total = weights[0];
    for (var i = 1; i < 5; i++) {
        for (var s = -1; s <= 1; s += 2) {
            let t = clamp(p + dir * (i * s), vec2<i32>(0), last);
            let w = weights[i] * same_surface(dist, texel_distance(t));
            sum += textureLoad(src, t, 0).r * w;
            total += w;
        }
    }
    return vec4(vec3(sum / total), 1.0);
}

@fragment
fn fs_ao_blur_h(in: VertexOutput) -> @location(0) vec4<f32> {
    return blur(vec2<i32>(in.pos.xy), vec2(1, 0));
}

@fragment
fn fs_ao_blur_v(in: VertexOutput) -> @location(0) vec4<f32> {
    return blur(vec2<i32>(in.pos.xy), vec2(0, 1));
}

// Half an 8-bit step of noise, as fs_ao_apply: the shade multiplied into the
// 8-bit scene otherwise shows as bands.
fn dither(pos: vec2<f32>) -> f32 {
    return (fract(52.9829189 * fract(dot(pos, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
}

// How much of the pixel the effects drawn over it make up (effect_mask_target()):
// their colour over black, its brightest channel times effect_gain. That much of
// the pixel is left unshaded. Only near the camera (dist, the distance of the
// surface behind, under 3,000 units and fading out by 6,000): far off, what is
// drawn so is the haze over the sea toward the horizon, which the cloud shadows
// shade with the sea.
fn effect_cover(uv: vec2<f32>, dist: f32) -> f32 {
    let size = textureDimensions(effects);
    let p = min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u));
    let c = textureLoad(effects, p, 0).rgb;
    return clamp(max(c.r, max(c.g, c.b)) * effect_gain, 0.0, 1.0) * (1.0 - smoothstep(3000.0, 6000.0, dist));
}

// To the scene's size: the four nearest texels, as a bilinear filter weighs
// them, each by how near its distance is to the pixel's. Where none is on the
// pixel's surface (a thin edge), the nearest in distance.
@fragment
fn fs_ao_upscale(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = distance_at(in.uv);
    if (dist <= 0.0) {
        return vec4(1.0);
    }
    let f = in.uv * vec2<f32>(textureDimensions(src)) - 0.5;
    let base = vec2<i32>(floor(f));
    let within = f - floor(f);
    let last = vec2<i32>(textureDimensions(src)) - vec2<i32>(1);
    var sum = 0.0;
    var total = 0.0;
    var closest = 1.0;
    var gap = 3.4e38;
    for (var i = 0; i < 4; i++) {
        let o = vec2(i % 2, i / 2);
        let t = clamp(base + o, vec2<i32>(0), last);
        let shade = textureLoad(src, t, 0).r;
        let other = texel_distance(t);
        let bilinear = select(1.0 - within.x, within.x, o.x == 1) * select(1.0 - within.y, within.y, o.y == 1);
        let w = bilinear * same_surface(dist, other);
        sum += shade * w;
        total += w;
        if (other > 0.0 && abs(other - dist) < gap) {
            gap = abs(other - dist);
            closest = shade;
        }
    }
    // Not on an effect drawn over the surface (effect_cover()).
    let shade = mix(select(closest, sum / total, total > 1e-3), 1.0, effect_cover(in.uv, 0.0));
    // DOL_AURORA_AO_DEBUG=2: what the occlusion darkens, tinted red over the
    // scene (four times as strong, so a faint shade shows), as the cascades are.
    if (debug_view > 1.5) {
        return vec4(mix(vec3(1.0), vec3(1.0, 0.15, 0.15), clamp((1.0 - shade) * 4.0, 0.0, 1.0)), 1.0);
    }
    return vec4(vec3(shade + select(0.0, dither(in.pos.xy), shade < 0.999)), 1.0);
}
)""";
wgpu::ShaderModule g_aoModule;

// Cloud shadows on the sea and the islands: each pixel's place in the world
// (from depth, or where the view ray meets the sea where there is no depth),
// darkened by soft drifting noise in world X/Z that fades with distance.
// The cloud shadows' noise, for their own pass (kCloudShader) and the sun shadows' cloud shadows on Wind
// Waker HD's rule (kSunMapShader); put before each.
constexpr const char* kCloudNoise = R"""(
// A gradient for each lattice point, from its integer coordinates: no sin(), so
// every GPU gets the same values whatever its precision.
fn gradient(p: vec2<f32>) -> vec2<f32> {
    var q = bitcast<vec2<u32>>(vec2<i32>(p));
    q = q * 1664525u + 1013904223u;
    q.x += q.y * 1664525u;
    q.y += q.x * 1664525u;
    q = q ^ (q >> vec2(16u));
    q.x += q.y * 1664525u;
    q.y += q.x * 1664525u;
    q = q ^ (q >> vec2(16u));
    let a = f32(q.x) * (6.2831853 / 4294967296.0);
    return vec2(cos(a), sin(a));
}

// Gradient noise, about 0 to 1. Value noise put the clouds' edges along the
// world's X and Z (square-looking shadows); gradient noise has no such rows.
fn gradient_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = dot(gradient(i), f);
    let b = dot(gradient(i + vec2(1.0, 0.0)), f - vec2(1.0, 0.0));
    let c = dot(gradient(i + vec2(0.0, 1.0)), f - vec2(0.0, 1.0));
    let e = dot(gradient(i + vec2(1.0, 1.0)), f - vec2(1.0, 1.0));
    return 0.5 + 0.9 * mix(mix(a, b, u.x), mix(c, e, u.x), u.y);
}

fn fbm(p0: vec2<f32>) -> f32 {
    // Each octave turned by about 37 degrees, so no two line up.
    let turn = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    var p = p0;
    var v = 0.0;
    var a = 0.5;
    for (var i = 0; i < 4; i++) {
        v += a * gradient_noise(p);
        p = turn * p * 2.03 + vec2(17.0, 9.0);
        a *= 0.5;
    }
    return v;
}
)""";

constexpr const char* kCloudShader = R"""(
override strength: f32 = 0.28;
override scale: f32 = 7000.0;
override debug_view: bool = false;
override effect_gain: f32 = 2.0;

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds
};

@group(0) @binding(0) var samp: sampler;
// The exposure's sea mask; for the cloud shadows, the effects' mask, which they leave.
@group(0) @binding(1) var mask: texture_2d<f32>;
@group(0) @binding(4) var scene: texture_2d<f32>;    // the scene before the effects, for the exposure
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;

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

// Half an 8-bit step of noise (interleaved gradient noise), so a smooth shade
// multiplied into the 8-bit scene doesn't come out as bands on the sea.
fn dither(pos: vec2<f32>) -> f32 {
    return (fract(52.9829189 * fract(dot(pos, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
}

// How much of the pixel the effects drawn over it make up (effect_mask_target()):
// their colour over black, its brightest channel times effect_gain. That much of
// the pixel is left unshaded. Only near the camera (dist, the distance of the
// surface behind, under 3,000 units and fading out by 6,000): far off, what is
// drawn so is the haze over the sea toward the horizon, which the cloud shadows
// shade with the sea.
fn effect_cover(uv: vec2<f32>, dist: f32) -> f32 {
    let size = textureDimensions(mask);
    let p = min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u));
    let c = textureLoad(mask, p, 0).rgb;
    return clamp(max(c.r, max(c.g, c.b)) * effect_gain, 0.0, 1.0) * (1.0 - smoothstep(3000.0, 6000.0, dist));
}

@fragment
fn fs_clouds(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureLoad(depth, vec2<i32>(in.pos.xy), 0);
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let ray = vec3(ndc.x * view.lens.x * view.lens.y, ndc.y * view.lens.x, -1.0);
    let dir = vec3(dot(view.rows[0].xyz, ray), dot(view.rows[1].xyz, ray), dot(view.rows[2].xyz, ray));
    let eye = vec3(view.rows[0].w, view.rows[1].w, view.rows[2].w);
    // Where the view ray meets the sea, if it does.
    var sea = -1.0;
    if (dir.y < -1e-4) {
        sea = (view.lens.w - eye.y) / dir.y;
    }
    var world = eye;
    if (d > 0.0 && (sea <= 0.0 || view.lens.z / d < 0.9 * sea)) {
        // Something in front of the sea (an island, a boat): its place from depth.
        world = eye + dir * (view.lens.z / d);
    } else {
        // The sea, also where it has no depth: its place from its plane. Taking
        // the sea's distance from depth too gave straight-edged steps far out
        // wherever depth is less precise (seen on Vulkan).
        if (sea <= 0.0) {
            return vec4(1.0);
        }
        world = eye + dir * sea;
    }
    let distance = length(world - eye);
    if (debug_view) {
        // A 1,000-unit world grid in red and green, height in blue (every 500 units).
        let g = fract(world.xz / 1000.0);
        return vec4(g.x, g.y, fract(world.y / 500.0), 1.0);
    }
    let drift = vec2(0.6, 0.25) * view.time.x * 0.02;
    let n = fbm(world.xz / scale + drift);
    // Not on an effect drawn over the surface (effect_cover()).
    let cover = effect_cover(in.uv, distance);
    let shade = smoothstep(0.42, 0.6, n) * strength * (1.0 - smoothstep(60000.0, 160000.0, distance)) *
                (1.0 - cover);
    return vec4(vec3(1.0 - shade + select(0.0, dither(in.pos.xy), shade > 0.0)), 1.0);
}

// DOL_AURORA_SEA_MASK_DEBUG=1: the sea mask over the scene instead (local debug).
override sea_debug: bool = false;
// Fading out from far_start to far_end units away (DOL_AURORA_EXPOSURE_FAR=start,end):
// what is that far is the haze's colour, and brightened it stood out pale cyan
// against the sky (the islands seen from the Private Oasis).
override far_start: f32 = 10000.0;
override far_end: f32 = 25000.0;
// What is already bright takes less of it (DOL_AURORA_EXPOSURE_HIGHLIGHTS=from,to):
// the gain tapers from full where a colour's brightest channel is `from` to
// none at `to`, so ice and white rock keep their shading (at 1.5 times they
// went flat white, a sixth of the screen clipped, at Ice Ring Isle), and the
// taper is gentle enough that a brighter colour still comes out brighter.
override highlight_from: f32 = 0.4;
override highlight_to: f32 = 1.0;
// The game's brightness (kBoost: scene * (1 + this) - the scene), carried in
// time.x of this pass's view, where there is depth and not the sea's surface
// (the sea mask): Wind Waker HD brightens the sun's light on actors and
// scenery, not on the sea or the sky. A beach below a swell's height takes it,
// and the swell doesn't.
@fragment
fn fs_exposure(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy);
    let d = textureLoad(depth, p, 0);
    if (d <= 0.0) {
        return vec4(0.0); // no depth: left as it is
    }
    let land = 1.0 - textureLoad(mask, p, 0).r;
    if (sea_debug) {
        // The sea mask: the sea twice as bright, the rest left as it is (the pass scales the scene by
        // one more than this; the target clamps what is below zero).
        return vec4(select(vec3(0.0), vec3(1.0), land < 0.5), 0.0);
    }
    // Wind Waker HD's character light (time.y 1 in this pass) has the brightness in it: the characters drawn with it
    // (the mask's alpha 1) are left as drawn.
    if (view.time.y > 0.5 && textureLoad(mask, p, 0).a > 0.5) {
        return vec4(0.0);
    }
    let near = 1.0 - smoothstep(far_start, far_end, view.lens.z / d);
    var gain = (view.time.x - 1.0) * land * near;
    let c = textureLoad(scene, p, 0).rgb;
    gain *= 1.0 - smoothstep(highlight_from, highlight_to, max(max(c.r, c.g), c.b));
    return vec4(vec3(select(0.0, gain + dither(in.pos.xy), gain > 0.0)), 0.0);
}
)""";
wgpu::ShaderModule g_cloudModule;

// Sun shadows from what is on screen: from each pixel's place in view space,
// steps toward the sun; a step that lands behind the depth buffer's surface (by
// less than an object's assumed thickness) is in shadow. Off-screen casters
// cast nothing; the sun low or below the horizon casts nothing. Drawn at a
// quarter of the size and blurred (the jittered steps' grain becomes a soft
// edge), then multiplied into the scene.
constexpr const char* kSunShader = R"""(
override strength: f32 = 0.35;
override reach: f32 = 1500.0;
override effect_gain: f32 = 2.0;

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds, then the direction to the sun (world)
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var effects: texture_2d<f32>; // the effects' mask, which the shade leaves
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;

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

fn view_depth(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth));
    let p = clamp(vec2<i32>(uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let d = textureLoad(depth, p, 0);
    return select(view.lens.z / d, 0.0, d <= 0.0);
}

// How much of the pixel the effects drawn over it make up (effect_mask_target()):
// their colour over black, its brightest channel times effect_gain. That much of
// the pixel is left unshaded. Only near the camera (dist, the distance of the
// surface behind, under 3,000 units and fading out by 6,000): far off, what is
// drawn so is the haze over the sea toward the horizon, which the cloud shadows
// shade with the sea.
fn effect_cover(uv: vec2<f32>, dist: f32) -> f32 {
    let size = textureDimensions(effects);
    let p = min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u));
    let c = textureLoad(effects, p, 0).rgb;
    return clamp(max(c.r, max(c.g, c.b)) * effect_gain, 0.0, 1.0) * (1.0 - smoothstep(3000.0, 6000.0, dist));
}

@fragment
fn fs_sun(in: VertexOutput) -> @location(0) vec4<f32> {
    let z = view_depth(in.uv);
    let sun = view.time.yzw;
    if (z <= 0.0 || sun.y < 0.05) {
        return vec4(1.0);
    }
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let tx = view.lens.x * view.lens.y;
    let ty = view.lens.x;
    let p = vec3(ndc.x * tx, ndc.y * ty, -1.0) * z;
    // The sun in view space: the inverse view's rotation transposed.
    let l = normalize(vec3(dot(vec3(view.rows[0].x, view.rows[1].x, view.rows[2].x), sun),
                           dot(vec3(view.rows[0].y, view.rows[1].y, view.rows[2].y), sun),
                           dot(vec3(view.rows[0].z, view.rows[1].z, view.rows[2].z), sun)));
    let jitter = fract(sin(dot(in.pos.xy, vec2(12.9898, 78.233))) * 43758.5453);
    var lit = 1.0;
    for (var i = 0; i < 24; i++) {
        let t = 20.0 + (reach - 20.0) * (f32(i) + jitter) / 24.0;
        let q = p + l * t;
        if (-q.z <= view.lens.z) {
            break;
        }
        let s = vec2(q.x / -q.z / tx, q.y / -q.z / ty);
        let uv = vec2(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        let surface = view_depth(uv);
        let behind = -q.z - surface;
        if (surface > 0.0 && behind > 12.0 + 0.004 * surface && behind < 250.0) {
            lit = 0.0;
            break;
        }
    }
    let fade = (1.0 - smoothstep(4000.0, 9000.0, z)) * smoothstep(0.05, 0.25, sun.y) * (1.0 - effect_cover(in.uv, z));
    return vec4(vec3(1.0 - strength * (1.0 - lit) * fade), 1.0);
}
)""";
wgpu::ShaderModule g_sunModule;

// Sun shadows from the shadow map: each pixel's place in the world, taken to
// the map (view.light), is in shadow where something nearer the sun was drawn
// there (nine filtered lookups around it, for a soft edge). The map fades out
// toward its edge. Alpha-tested draws are not in the map (leaves, fences), so
// a few of the screen's steps (as fs_sun, out to `contact`) could still look
// for them (screen_steps); they are off unless asked for. Multiplied straight
// into the scene at its full size (direct), or drawn at a quarter of the size
// for the debug view.
constexpr const char* kSunMapShader = R"""(
override strength: f32 = 0.35;
override contact: f32 = 300.0;
override debug_view: f32 = 0.0;
override screen_steps: f32 = 0.0;
override normal_offset: f32 = 2.0;
override slope_texels: f32 = 20.0;
override slope_near: f32 = 20.0;
override hd_bias: f32 = 0.0;
// Wind Waker HD's terrain mix (hd_mix): low_pass writes the map's shadow and HD's grazing term at a quarter of
// the size (blurred after), the main pass mixes the blurred one in by distance and grazing and takes HD's curve.
override low_pass: f32 = 0.0;
override hd_mix: f32 = 0.0;
override mix_e_start: f32 = 545.0;
override mix_e_near: f32 = 4.0; // the curve's power near the camera (HD's 4; 2 farther off)
// Wind Waker HD's shadow pass (hd_prepass, the "HD" shadow quality): the low pass is HD's pre-pass, at half the
// size, one hardware-filtered lookup per pixel in the cascade its distance picks (view.hd: HD's 420 and 1395),
// blurred; the main pass takes every pixel's shadow from it, as HD's materials read its pre-pass.
override hd_prepass: f32 = 0.0;
// The mask's green from which a pixel is a character (keeps its own shading, BlueWake's bias, no facing term or
// terrain mix): 0.5 takes the skinned (green 0.6) as well as the toon-shaded (1); 0.8 the toon-shaded alone
// (g_sunSkinnedScenery), as Wind Waker HD draws Outset's palms with its scenery's shader.
override character_g: f32 = 0.5;
override sea_shadow: f32 = 1.0; // the sea's shadow strength (HD's cPrimShadowIntensity: 0.5 in HD's pass)
override mix_blur_start: f32 = 1318.0;
override mix_blur_end: f32 = 1833.0;
override normal_near: f32 = 2.0;
override far_start: f32 = 1500.0;
override far_end: f32 = 5000.0;
override direct: f32 = 1.0;
override spread: f32 = 4.0;
override smooth_terms: f32 = 1.0;
override sea_low: f32 = 40.0;
override gap_fill: f32 = 1.0;
override taps: f32 = 12.0;
override sea_rise: f32 = 0.0;
override sea_bias: f32 = 0.0;
override soft: f32 = 0.01;
override facing_on: f32 = 0.0;
override facing_mul: f32 = 10.0;
override facing_depth_mul: f32 = 0.0;
override facing_depth_blur: f32 = 8.0;
override smooth_ambient: f32 = 0.0;
override prepass_far: f32 = 0.0;
override facing_far_start: f32 = 12000.0;
override facing_far_end: f32 = 14000.0;
override shape_on: f32 = 1.0;
override shape_near: f32 = 300.0;
override shape_far: f32 = 500.0;
override soft_max: f32 = 4.0;
override effect_gain: f32 = 2.0;
override effect_debug: f32 = 0.0;
// Wind Waker HD's SSAO (g_aoHd, with hd_mix): the low pass writes it in blue, the main pass darkens the shade with
// it. Its values (DOL_AURORA_AO_HD_*) are HD's: radius 0.03 of the screen's height, gone at 10,000 units, what is
// over 320 units nearer left out, density 1.1, the centre's depth x (1 - 0.001).
override ao_radius: f32 = 0.03;
override ao_far: f32 = 10000.0;
override ao_attn: f32 = 320.0;
override ao_density: f32 = 1.1;
override ao_offset: f32 = 0.001;
override ao_power: f32 = 3.0;  // the blurred AO to this power, for HD's sharp x blurred squared
override ao_facing: f32 = 1.0; // HD's fade on faces square to the camera, pow(N.z, 20)
override ao_debug: f32 = 0.0;  // 1: the AO as it darkens the shade, alone
override cloud_scale: f32 = 7000.0; // the cloud shadows on HD's rule (g_cloudHd): world units across one cloud
override cloud_repeat: f32 = 15000.0;
override cloud_speed: f32 = 0.01116;

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds, then the direction to the sun (world)
    light: array<vec4<f32>, 12>, // the world to each cascade (post_fx::SunMap::world), three rows each
    params: vec4<f32>,          // a texel's size in the map, the cascades drawn, how strongly the shadows show, near / (far - near)
    bias: vec4<f32>,            // each cascade's depth bias, in its own depth's units
    facing: vec4<f32>,          // x: 1 when `mask` is this frame's sea mask (the facing term needs it); y: 1 with HD's palette;
                                // z: 1 when `normals` holds the scenery's smooth normals; w: 1 with HD's SSAO
    shade: vec4<f32>,           // the full shadow's colour the scene is multiplied by; with HD's palette, its sky light (linear);
                                // w: 1 with the cloud shadows on HD's rule
    palette: vec4<f32>,         // with HD's palette: the ambient its shade takes from the side (linear); w: the cloud
                                // shadows' floor (HD's cCloudShadowPower)
    lit: vec4<f32>,             // with HD's palette: its sunlight (linear); w: 1 with HD's cloud texture bound
    hd: vec4<f32>,              // HD's pass: x, y the distances from the camera where it moves to the second and third
                                // cascade (HD's 420 and 1395, times the camera's zoom); z: 1 when the characters took
                                // Wind Waker HD's light in their own draws (post_fx::actor_light_applied())
    fit: vec4<f32>,             // x: 1 when the cascades follow the camera (each fitted to a slice of the view
                                // between splits); y: where the fourth starts (0: three); z, w: the far fade's start
                                // and end (lit from there, HD's pre-pass's far fade)
    points: vec4<f32>,          // Wind Waker HD's light buffer on the lit scenery: x how many lights, y the scenery's
                                // weight (HD's 1.25), z 1 when the scenery took them in its draws
    pl: array<vec4<f32>, 64>,   // each light: its world position and radius, then its linear colour
    fog: vec4<f32>,             // the lit scenery's fog as its draws took it (GX's): A, B's magnitude, B's shift, C
    fog_colour: vec4<f32>,      // its colour (0-1); w: 0 none, 1 linear, 2 HD's curve (its square)
    fog_x: vec4<f32>,           // GX's range adjust: x 1 with it, y the centre, z the width
    fog_range: array<vec4<f32>, 3>, // and its table (10 values)
    fog_starts: array<vec4<f32>, 4>, // HD's palette entries with a fog start below 0: start, end pairs (end 0: none)
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var mask: texture_2d<f32>; // the sea mask: red the sea, green a character
@group(0) @binding(10) var effects: texture_2d<f32>; // the effects' mask, which the shade leaves
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var sun_map: texture_depth_2d_array;
@group(0) @binding(5) var sun_compare: sampler_comparison;
// The scenery's smooth normals (view space x 0.5 + 0.5, alpha 1 where a draw gave one; gxcore_draw.cpp).
@group(0) @binding(6) var normals: texture_2d<f32>;
// The blurred quarter-size shadow and grazing term (hd_mix; low_pass writes it before the blur).
@group(0) @binding(7) var shadow_blur: texture_2d<f32>;
// Wind Waker HD's cloud shadow texture (cloud_tex), repeating.
@group(0) @binding(8) var cloud_map: texture_2d<f32>;
@group(0) @binding(9) var cloud_samp: sampler;

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

fn view_depth(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth));
    let p = clamp(vec2<i32>(uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let d = textureLoad(depth, p, 0);
    // near / depth is the distance only for a far plane infinitely far; with
    // the camera's own (params.w: near / (far - near)) it is exact.
    return select(view.lens.z / (d + view.params.w), 0.0, d <= 0.0);
}

// How much of the pixel the effects drawn over it make up (effect_mask_target()):
// their colour over black, its brightest channel times effect_gain. That much of
// the pixel is left unshaded. Only near the camera (dist, the distance of the
// surface behind, under 3,000 units and fading out by 6,000): far off, what is
// drawn so is the haze over the sea toward the horizon, which the cloud shadows
// shade with the sea.
fn effect_cover(uv: vec2<f32>, dist: f32) -> f32 {
    let size = textureDimensions(effects);
    let p = min(vec2<u32>(uv * vec2<f32>(size)), size - vec2<u32>(1u));
    let c = textureLoad(effects, p, 0).rgb;
    return clamp(max(c.r, max(c.g, c.b)) * effect_gain, 0.0, 1.0) * (1.0 - smoothstep(3000.0, 6000.0, dist));
}

// The farthest of the four depth texels around uv (a half-size pass's pixel centre is their shared corner), as HD's
// pre-pass gathers them; the sky (0) if any of them is.
fn view_depth_far(uv: vec2<f32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth));
    let base = vec2<i32>(floor(uv * vec2<f32>(size) - 0.5));
    var d = 1.0;
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let p = clamp(base + vec2(i, j), vec2<i32>(0), size - vec2<i32>(1));
            d = min(d, textureLoad(depth, p, 0));
        }
    }
    return select(view.lens.z / (d + view.params.w), 0.0, d <= 0.0);
}

// Wind Waker HD's SSAO (agl's ssao.sh, PASS_AO_BUFFER with 4 pairs, as Cemu's dump of it computes; its sample
// pairs, weights and 4x4 rotation tile read from a RenderDoc capture). HD's depth is normalised to its far plane
// (400,000). For each pair, a sample in front of the centre occludes (all but at once: the depth difference is
// divided by the radius in screen units); one over ao_attn units in front is replaced by its mirror's opposite.
// Flat ground sums to 0.5, which is no darkening.
const kAoPoints = array<vec4<f32>, 4>(
    vec4(0.278512, 0.340634, 1.11359, 0.14793), vec4(0.238801, 0.02396, 1.03011, 0.107181),
    vec4(-0.0638933, 0.636803, 1.30145, 0.115841), vec4(-0.6503, 0.531704, 1.84302, 0.0967856));
const kAoRotate = array<vec2<f32>, 16>(
    vec2(0.9216, 0.3804), vec2(-0.8353, 0.5529), vec2(-0.2000, 0.9765), vec2(0.7020, 0.7020),
    vec2(0.1922, 0.9765), vec2(0.5529, 0.8275), vec2(1.0000, -0.0039), vec2(-0.3882, 0.9216),
    vec2(-0.7098, 0.7020), vec2(0.3804, 0.9216), vec2(0.9765, 0.1922), vec2(-0.9843, 0.1922),
    vec2(-0.9294, 0.3804), vec2(-0.0039, 1.0000), vec2(0.8275, 0.5529), vec2(-0.5608, 0.8275));
fn ao_depth(uv: vec2<f32>) -> f32 {
    let s = view_depth(uv);
    return select(1.0, s / 400000.0, s > 0.0); // no depth (the sky): the far plane
}
fn hd_ao(uv: vec2<f32>, z: f32, pix: vec2<f32>) -> f32 {
    let r = ao_radius * clamp(1.0 - z / ao_far, 0.0, 1.0);
    if (r <= 0.0) {
        return 1.0;
    }
    let ip = vec2<u32>(pix) % vec2(4u);
    let rot = kAoRotate[ip.y * 4u + ip.x];
    let inv_aspect = 1.0 / view.lens.y;
    let c = z / 400000.0 * (1.0 - ao_offset);
    let k = 399997.0 / r;
    let attn = 400000.0 / ao_attn;
    var total = 0.0322618;
    for (var i = 0; i < 4; i++) {
        let sp = kAoPoints[i];
        let q = sp.xy * r;
        let o = vec2((rot.x * q.x - rot.y * q.y) * inv_aspect, rot.y * q.x + rot.x * q.y);
        let dp = c - ao_depth(uv + o);
        let dm = c - ao_depth(uv - o);
        let op = clamp(dp * k * 0.5 * sp.z + 0.5, 0.0, 1.0);
        let om = clamp(dm * k * 0.5 * sp.z + 0.5, 0.0, 1.0);
        let ap = clamp(1.0 - dp * attn, 0.0, 1.0);
        let am = clamp(1.0 - dm * attn, 0.0, 1.0);
        let fp = mix(0.5 + (0.5 - om) * am, op, ap);
        let fm = mix(0.5 + (0.5 - op) * ap, om, am);
        total += (fp + fm) * sp.w;
    }
    return 1.0 - clamp((total - 0.5) * ao_density * 2.0, 0.0, 1.0);
}

// Whether the surface a screen step found (at s, its depth) is one the map
// holds: its depth from the sun is the map's there. Such a caster casts its
// own shadow from the map; Link, seen in front of the ground just behind him,
// also darkened a band of it in his shape.
fn in_sun_map(s: vec2<f32>, surface: f32, tx: f32, ty: f32) -> bool {
    let o = vec4(vec3(s.x * tx, s.y * ty, -1.0) * surface, 1.0);
    let w = vec4(dot(view.rows[0], o), dot(view.rows[1], o), dot(view.rows[2], o), 1.0);
    let k = first_cascade(w);
    if (k < 0) {
        return false;
    }
    let m = to_cascade(k, w);
    let size = vec2<f32>(textureDimensions(sun_map));
    let texel = clamp(vec2<i32>(vec2(m.x * 0.5 + 0.5, 0.5 - m.y * 0.5) * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    return abs(textureLoad(sun_map, texel, k, 0) - m.z) < 8.0 * view.bias[k];
}

// A place in the world in cascade k: x and y -1 to 1 across it, z 0 to 1.
fn to_cascade(k: i32, w: vec4<f32>) -> vec3<f32> {
    return vec3(dot(view.light[k * 3], w), dot(view.light[k * 3 + 1], w), dot(view.light[k * 3 + 2], w));
}

// How far out a place is in cascade k (m: its place there), 1 at the edge:
// across the sun's view and along the sun's rays alike, in the cascade's range.
// Measured across only, the sea far below Link along the sun's rays from him
// (where the platform's shadow falls) was in the sharpest cascade, a square of
// other terms amid the coarser cascades' water.
fn reach(k: i32, m: vec3<f32>) -> f32 {
    // Fitted to the view's slices (view.fit.x), a cascade holds all of its slice along the sun.
    let along = select(abs(m.z - 0.5) * length(view.light[k * 3].xyz) / length(view.light[k * 3 + 2].xyz), 0.0,
                       view.fit.x > 0.5);
    return max(max(abs(m.x), abs(m.y)), along);
}

fn inside(k: i32, m: vec3<f32>) -> bool {
    return reach(k, m) < 1.0 && m.z > 0.0 && m.z < 1.0;
}

// A place moved out along its surface's normal by about two of cascade k's
// texels (the reach of its filter), so a coarse cascade doesn't shade the
// surface it was drawn from: on a beach under a low sun the ground's own depth
// from the sun changes more across a texel than the depth bias allows.
fn offset_world(k: i32, w: vec4<f32>, normal: vec3<f32>) -> vec4<f32> {
    let texel = 2.0 / (min(length(view.light[k * 3].xyz), length(view.light[k * 3 + 1].xyz)) * f32(textureDimensions(sun_map).x));
    return vec4(w.xyz + normal * texel, 1.0);
}

// How much cascade k's bias, normal offset, slope term and filter reach are
// scaled up at a place: a continuous function of how far out it is, so they
// grow by `spread` from one cascade's inner edge to its outer one and match the
// next cascade's where the two meet. With a step at each edge, the sea's own
// crest shading stopped there in a square around Link. In cascade k, `t` runs
// 0 at the edge of cascade k-1 to 1 at k's own; in the first, from a quarter
// of the way out (Link's surroundings keep the near terms).
fn grow_at(k: i32, w: vec4<f32>) -> f32 {
    if (smooth_terms < 0.5) {
        return 1.0;
    }
    let mk = to_cascade(k, w);
    let outer = reach(k, mk);
    var t = 0.0;
    if (k == 0) {
        t = clamp((outer - 0.25) / 0.75, 0.0, 1.0);
    } else {
        let mi = to_cascade(k - 1, w);
        let inner = reach(k - 1, mi);
        // inner >= 1 outside cascade k-1, outer <= 1 inside k.
        t = clamp((inner - 1.0) / max((inner - 1.0) + (1.0 - outer), 1e-4), 0.0, 1.0);
    }
    return pow(spread, t);
}

// HD's cascade for a place z units from the camera: the first past each split (no blending), or a later one if it
// doesn't hold the place; -1 when none does.
fn hd_cascade(z: f32, w: vec4<f32>) -> i32 {
    var k = select(0, 1, z > view.hd.x) + select(0, 1, z > view.hd.y)
        + select(0, 1, view.fit.x > 0.5 && view.fit.y > 0.0 && z > view.fit.y);
    if (view.fit.x > 0.5 && z >= view.fit.w) {
        return -1;
    }
    for (; k < i32(view.params.y); k++) {
        if (inside(k, to_cascade(k, w))) {
            return k;
        }
    }
    return -1;
}

// The sharpest cascade that holds a place, or -1.
fn first_cascade(w: vec4<f32>) -> i32 {
    for (var k = 0; k < i32(view.params.y); k++) {
        if (inside(k, to_cascade(k, w))) {
            return k;
        }
    }
    return -1;
}

// How lit a place is in cascade k (m: its place there): a disc of lookups two
// texels wide, turned per pixel (stable, from the pixel alone), so the edge
// comes out soft and the map's texels break into fine noise instead of stairs.
fn lit_in(k: i32, m: vec3<f32>, rot: mat2x2<f32>, slope: f32, world_y: f32, grow: f32) -> f32 {
    let uv = vec2(m.x * 0.5 + 0.5, 0.5 - m.y * 0.5);
    // The surface's own depth from the sun changes by `slope` world units per
    // unit across the map: over the filter's reach that is how much farther a
    // lit surface can read than its neighbours, so the bias grows with it
    // (`slope` already holds how many texels of it, more far from the camera).
    // With the sun low over flat ground (20 degrees at Outset in the evening)
    // the ground shaded itself, and the sea in rings, without it.
    let size = vec2<f32>(textureDimensions(sun_map));
    let texel = 2.0 / (min(length(view.light[k * 3].xyz), length(view.light[k * 3 + 1].xyz)) * size.x);
    let per_unit = length(view.light[k * 3 + 2].xyz);  // the map's depth per world unit along the sun
    let reach_bias = texel * slope * per_unit * grow;  // the slope term, which grows with the filter's reach
    let base_reference = m.z - view.bias[k] * grow - reach_bias;
    // HD's pre-pass: one hardware-filtered lookup (its two jittered taps fall on one place: cBias is 0). The
    // scenery's slope term is 0 with HD's bias; characters keep BlueWake's (they cast with BlueWake's bias, and
    // without it Link shaded all of himself).
    if (hd_prepass > 0.5) {
        return textureSampleCompareLevel(sun_map, sun_compare, uv, k, base_reference);
    }
    // Nothing at the sea's height or below shades the water or the sand at its
    // edge: the islands' models carry their sea floor up to the waterline, and
    // the waves dip below it, so the sea read it as something above.
    // How much the surface counts as sea: fully below sea_low over the sea's
    // height, not at all 40 units higher (no hard line for the waves to cross).
    let low = 1.0 - smoothstep(sea_low, sea_low + 40.0, world_y - view.lens.w);
    // How far the caster is, along the sun's rays: a few lookups over the
    // widest the filter may get, averaging what stands in the way. The
    // filter widens with it, so a long shadow's tip is soft, as in Wind Waker
    // HD. (Growing the bias with it too moved shadows along the ground as the
    // camera neared an island, since the widening changes with distance.
    // Fading the shadow with the caster's distance as well was tried:
    // where two shadows overlap the map keeps only the caster nearer the sun,
    // so the overlap came out lighter than the shadow under it.) Where no
    // lookup finds anything in the way the place is lit, and the filter's own
    // lookups are skipped.
    var widen = 1.0;
    let reference = base_reference;
    if (soft > 0.0) {
        let probes = max(4, n_probes());
        var found = 0.0;
        var distance = 0.0;
        for (var i = 0; i < probes; i++) {
            let r = 2.0 * (1.0 + soft_max) * sqrt((f32(i) + 0.5) / f32(probes));
            let a = 2.3999632 * f32(i) + 1.2;
            let o = rot * (vec2(cos(a), sin(a)) * r) * view.params.x * grow;
            let t = clamp(vec2<i32>((uv + o) * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
            let d = textureLoad(sun_map, t, k, 0);
            if (d < base_reference) {
                found += 1.0;
                distance += (m.z - d) / per_unit;
            }
        }
        if (found < 0.5 && !(low > 0.0 && sea_rise > 0.0)) {
            return 1.0;
        }
        widen = 1.0 + min(soft * distance / max(found, 1.0) / (2.0 * texel * grow), soft_max);
    }
    var sum = 0.0;
    let n = i32(taps);
    for (var i = 0; i < n; i++) {
        var o = vec2(0.0);
        if (i > 0) {
            let r = 2.0 * widen * sqrt((f32(i - 1) + 0.5) / (taps - 1.0));
            let a = 2.3999632 * f32(i - 1);
            o = rot * (vec2(cos(a), sin(a)) * r) * view.params.x * grow;
        }
        let t = clamp(vec2<i32>((uv + o) * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
        let d = textureLoad(sun_map, t, k, 0);
        var lit = select(0.0, 1.0, reference <= d);
        if (lit < 0.5 && low > 0.0 && sea_rise > 0.0) {
            // How high the surface in the way stands: up the sun's ray by
            // its distance in front. Lit below sea_rise over the sea, shaded
            // from 45 units higher, smoothly between: with a hard line, the
            // foot of a tower standing in the sea flickered as the waves
            // moved the water's height across it.
            // The rule's heights shrink with the sun's: what it leaves out casts
            // a strip rise / tan(elevation) long, so with fixed heights a
            // tower standing in the sea had its shadow start ever further from
            // its foot as the sun set (170 units at 5 degrees for 15 units).
            // Measured from the water here, not the sea's level: in a deep
            // sea's swell a tower's foot stood in troughs well below that
            // level, the part of it above the water counted as sea and its
            // shadow came away from the foot and back with every swell.
            let rise = view.time.z * (m.z - d) / per_unit;
            let scale = clamp(view.time.z * 2.0, 0.1, 1.0);
            lit = low * (1.0 - smoothstep(sea_rise * scale, (sea_rise + 45.0) * scale, rise));
        }
        sum += lit;
    }
    return sum / taps;
}

// The caster search's lookups: half the filter's.
fn n_probes() -> i32 {
    return i32(taps) / 2;
}

// A pixel's place in view space.
fn view_pos(uv: vec2<f32>, tx: f32, ty: f32) -> vec3<f32> {
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3(ndc.x * tx, ndc.y * ty, -1.0) * view_depth(uv);
}

// The surface's normal in view space from the depth around it, each way from the neighbour nearer in depth,
// so the edge of something in front doesn't take the slope down to what is behind it (the derivatives'
// normal does, and the facing term would outline every silhouette).
fn depth_normal(uv: vec2<f32>, p: vec3<f32>, tx: f32, ty: f32) -> vec3<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(depth));
    let l = view_pos(uv - vec2(px.x, 0.0), tx, ty);
    let r = view_pos(uv + vec2(px.x, 0.0), tx, ty);
    let u = view_pos(uv - vec2(0.0, px.y), tx, ty);
    let d = view_pos(uv + vec2(0.0, px.y), tx, ty);
    let dx = select(r - p, p - l, l.z < 0.0 && (r.z >= 0.0 || abs(p.z - l.z) < abs(r.z - p.z)));
    let dy = select(d - p, p - u, u.z < 0.0 && (d.z >= 0.0 || abs(p.z - u.z) < abs(d.z - p.z)));
    var n = cross(dx, dy);
    let len = length(n);
    n = select(vec3(0.0, 0.0, 1.0), n / len, len > 1e-12);
    return select(n, -n, dot(n, -p) < 0.0);
}

// The shape's own facing (the depth's normal, x facing_depth_mul), blurred as HD's shadow pre-pass blurs it:
// 3 x 3 points facing_depth_blur pixels apart (at 1080 lines, wider with distance), leaving out points whose
// depth jumps (another object, or the sky past an edge), so the shade doesn't cross a silhouette.
fn depth_facing_blurred(uv: vec2<f32>, z: f32, tx: f32, ty: f32, sun: vec3<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth));
    let stride = facing_depth_blur * (size.y / 1080.0) * mix(1.0, 3.0, smoothstep(1000.0, 8000.0, z)) / size;
    var sum = 0.0;
    var weight = 0.0;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let q = uv + vec2<f32>(f32(i), f32(j)) * stride;
            let pq = view_pos(q, tx, ty);
            if (abs(-pq.z - z) > 0.1 * z) {
                continue;
            }
            let nq = depth_normal(q, pq, tx, ty);
            let nw = vec3(dot(view.rows[0].xyz, nq), dot(view.rows[1].xyz, nq), dot(view.rows[2].xyz, nq));
            let w = (2.0 - abs(f32(i))) * (2.0 - abs(f32(j)));
            sum += w * clamp(dot(nw, sun) * facing_depth_mul, 0.0, 1.0);
            weight += w;
        }
    }
    return select(1.0, sum / weight, weight > 0.0);
}

// Wind Waker HD's light buffer at a world position and unit world normal (its light pass): the sum of
// colour x sat(N.L) x sat(1 - d / r)^1.5 (the lit scenery's draws add the same, gxcore_shader.cpp hd_point_sum).
fn point_sum(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var sum = vec3(0.0);
    let count = u32(view.points.x);
    for (var i = 0u; i < 32u; i++) {
        if (i >= count) {
            break;
        }
        let v = view.pl[i * 2u].xyz - p;
        let d = length(v);
        let fall = clamp(1.0 - d / view.pl[i * 2u].w, 0.0, 1.0);
        if (fall <= 0.0) {
            continue;
        }
        sum += view.pl[i * 2u + 1u].rgb * clamp(dot(n, v / max(d, 1e-4)), 0.0, 1.0) * pow(fall, 1.5);
    }
    return sum;
}

fn sun_map_shade(in: VertexOutput) -> vec4<f32> {
    let z = select(view_depth(in.uv), view_depth_far(in.uv), low_pass > 0.5 && hd_prepass > 0.5 && prepass_far > 0.5);
    let sun = view.time.yzw;
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let tx = view.lens.x * view.lens.y;
    let ty = view.lens.x;
    let p = vec3(ndc.x * tx, ndc.y * ty, -1.0) * z;
    let world = vec4(dot(view.rows[0], vec4(p, 1.0)), dot(view.rows[1], vec4(p, 1.0)), dot(view.rows[2], vec4(p, 1.0)), 1.0);
    // The surface's normal from its neighbours (before any early return: the
    // derivatives need every pixel), turned toward the camera.
    var normal = cross(dpdx(world.xyz), dpdy(world.xyz));
    // The map itself, straight to the screen: its depth from the sun as
    // contour bands, so the casters' silhouettes read regardless of range.
    // 2 shows the first cascade, 3 the second, 4 the third, 5 the fourth.
    if (debug_view > 1.5 && debug_view < 5.5) {
        let size = vec2<f32>(textureDimensions(sun_map));
        let p = clamp(vec2<i32>(in.uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
        let d = textureLoad(sun_map, p, min(i32(debug_view + 0.5) - 2, i32(view.params.y) - 1), 0);
        if (d >= 1.0) {
            return vec4(1.0);
        }
        return vec4(vec3(0.25 + 0.75 * fract(d * 100.0)), 1.0);
    }
    if (z <= 0.0 || view.params.z <= 0.0) {
        return vec4(1.0);
    }
    let nl = length(normal);
    normal = select(vec3(0.0), normal / nl, nl > 1e-6);
    let eye = vec3(view.rows[0].w, view.rows[1].w, view.rows[2].w);
    normal = select(normal, -normal, dot(normal, eye - world.xyz) < 0.0);
    // The lookup is moved out along a level normal on the sea: the waves'
    // own normals swing with every wave, and the lookup wobbled across the
    // edges of shadows on the water. The slope term keeps the waves' normals,
    // so a wave's face away from the sun doesn't shade itself (with a level
    // normal there too, the crests' shading boiled).
    let sea = 1.0 - smoothstep(sea_low, sea_low + 40.0, world.y - view.lens.w);
    var push = normal;
    if (sea > 0.0) {
        push = normalize(mix(normal, vec3(0.0, 1.0, 0.0), sea) + vec3(0.0, 1e-6, 0.0));
    }
    var lit = 1.0;
    // 0 near the camera, 1 far off: the bias terms grow with it.
    let far = smoothstep(far_start, far_end, z);
    // In HD's pass the main pass reads the pre-pass instead of the map (below), and the pre-pass picks HD's cascade.
    var k = first_cascade(world);
    // 6 (the cascades tinted) looks them up in the main pass too.
    if (hd_prepass > 0.5) {
        k = select(-1, hd_cascade(z, world), low_pass > 0.5 || (debug_view > 5.5 && debug_view < 6.5));
    }
    if (k >= 0) {
        let turn = 6.2831853 * fract(52.9829189 * fract(dot(in.pos.xy, vec2(0.06711056, 0.00583715))));
        let rot = mat2x2<f32>(cos(turn), sin(turn), -sin(turn), cos(turn));
        let grow = grow_at(k, world);
        // The sea casts nothing, so it can't shade itself and needs neither
        // the normal offset nor the slope bias: with them, a tower standing in
        // the water at a low sun had its shadow start well away from its foot.
        let unbiased = mix(1.0, sea_bias, sea);
        let offset = mix(normal_near, normal_offset, far) * unbiased;
        var m = to_cascade(k, offset_world(k, world, push * offset * grow));
        // tan of the angle between the surface and the sun (capped), times how
        // many texels of it the bias takes (the same near the camera as far
        // off by default: fewer near the camera let Link's side turned from
        // the sun shade itself).
        let ndl = dot(normal, sun);
        let cap = mix(4.0, 8.0, far);
        let tilt = select(cap, min(sqrt(max(1.0 - ndl * ndl, 0.0)) / ndl, cap), ndl > 0.0);
        var slope = tilt * mix(slope_near, slope_texels, far) * unbiased;
        // Wind Waker HD's bias instead (hd_bias; its shadow pre-pass): the lookup moved along the normal by
        // 20 x (1 - |N.L|)^9 world units, so only a surface nearly edge-on to the sun is pushed out, and no
        // slope term here: HD biases the casters' depth when drawing the map (gxcore_draw.cpp).
        // Characters (the sea mask's green) keep BlueWake's own bias: with HD's, Link shaded himself (HD shades
        // its characters in their own material, not through this).
        var character = false;
        if (view.facing.x > 0.5) {
            let csize = vec2<f32>(textureDimensions(mask));
            let cp = clamp(vec2<i32>(in.uv * csize), vec2<i32>(0), vec2<i32>(csize) - vec2<i32>(1));
            character = textureLoad(mask, cp, 0).g > character_g;
        }
        if (hd_bias > 0.5 && !character) {
            let graze = pow(clamp(1.0 - abs(ndl), 0.0, 1.0), 9.0);
            m = to_cascade(k, vec4(world.xyz + push * (20.0 * graze * unbiased), 1.0));
            slope = 0.0;
        }
        lit = lit_in(k, m, rot, slope, world.y, grow);
        // Near a cascade's edge, blend into the next (or, past the last, into
        // the light), so the step in sharpness doesn't show as a line.
        let edge = reach(k, m);
        let blend = select(smoothstep(0.85, 1.0, edge), 0.0, hd_prepass > 0.5);
        if (blend > 0.0) {
            var next = 1.0;
            if (k + 1 < i32(view.params.y)) {
                let grow_next = grow_at(k + 1, world);
                let n = to_cascade(k + 1, offset_world(k + 1, world, push * offset * grow_next));
                if (inside(k + 1, n)) {
                    next = lit_in(k + 1, n, rot, slope, world.y, grow_next);
                }
            }
            lit = mix(lit, next, blend);
        }
        // The bias leaves out what stands within its reach of a surface toward
        // the sun, so a shadow started that far from its caster's foot, and
        // further the further out (the bias grows with the camera's distance
        // and the cascade's texels): zoomed out, shadows came away from the
        // things that cast them. A few steps toward the sun through the depth
        // buffer, over just that reach, find the foot again. A surface a step
        // finds counts only if the map holds it, so what casts nothing (the
        // sea: a swell's crest dotted the sea with shade at sunrise) can't
        // cast here either, and the sea gets its feet back too.
        if (gap_fill > 0.5 && lit > 0.0) {
            let size = f32(textureDimensions(sun_map).x);
            let texel_world = 2.0 / (length(view.light[k * 3].xyz) * size);
            let per_unit = length(view.light[k * 3 + 2].xyz);
            let bias_world = (view.bias[k] / per_unit + texel_world * slope) * grow
                + texel_world * mix(normal_near, normal_offset, far) * grow;
            let reach = min(bias_world * 1.25, 800.0);
            if (reach > 4.0) {
                let l = normalize(vec3(dot(vec3(view.rows[0].x, view.rows[1].x, view.rows[2].x), sun),
                                       dot(vec3(view.rows[0].y, view.rows[1].y, view.rows[2].y), sun),
                                       dot(vec3(view.rows[0].z, view.rows[1].z, view.rows[2].z), sun)));
                let jitter = fract(52.9829189 * fract(dot(in.pos.xy, vec2(0.00583715, 0.06711056))));
                // Steps close enough together not to pass through a leg
                // (16 over the reach): with 6, Link's calves fell between
                // them and his feet's shadow stood apart from the rest.
                for (var i = 0; i < 16; i++) {
                    let t = reach * (f32(i) + jitter) / 16.0 + 2.0;
                    let q = p + l * t;
                    if (-q.z <= view.lens.z) {
                        break;
                    }
                    let sq = vec2(q.x / -q.z / tx, q.y / -q.z / ty);
                    let suv = vec2(sq.x * 0.5 + 0.5, 0.5 - sq.y * 0.5);
                    if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
                        break;
                    }
                    let surface = view_depth(suv);
                    let behind = -q.z - surface;
                    // Only just behind the surface found: a step deep behind
                    // something nearer the camera drew a copy of it, standing
                    // behind it, on whatever was further back.
                    // One-sided: what the map has nothing at (the sea's crests)
                    // casts nothing, but a surface the map holds something
                    // nearer the sun than counts (it is solid, in its own
                    // shadow): checked both ways, an island's far side, which
                    // the map holds only in patches far off, came out in
                    // blotches.
                    var held = true;
                    if (surface > 0.0 && behind > 2.0 + 0.002 * surface && behind < min(reach, 120.0)) {
                        let hp = vec4(vec3(sq.x * tx, sq.y * ty, -1.0) * surface, 1.0);
                        let hw = vec4(dot(view.rows[0], hp), dot(view.rows[1], hp), dot(view.rows[2], hp), 1.0);
                        let hm = to_cascade(k, hw);
                        if (inside(k, hm)) {
                            let hsize = vec2<f32>(textureDimensions(sun_map));
                            let huv = vec2(hm.x * 0.5 + 0.5, 0.5 - hm.y * 0.5);
                            let ht = clamp(vec2<i32>(huv * hsize), vec2<i32>(0), vec2<i32>(hsize) - vec2<i32>(1));
                            let hd = textureLoad(sun_map, ht, k, 0);
                            held = (hd - hm.z) / per_unit < max(40.0, texel_world * 4.0 * grow);
                        }
                    }
                    if (held && surface > 0.0 && behind > 2.0 + 0.002 * surface && behind < min(reach, 120.0)) {
                        // Full at the foot, where the map holds nothing, and
                        // fading toward the reach's end, where the map's own
                        // shadow takes over (at full strength there, the foot
                        // stood in a dark blob larger than the shadow).
                        lit = min(lit, smoothstep(0.7, 1.0, t / reach));
                        break;
                    }
                }
            }
        }
        // 7: how the surface faces the sun (green toward it, red away, white
        // edge-on) and, dimmed, how lit it came out.
        if (debug_view > 6.5 && debug_view < 7.5) {
            let nd = dot(normal, sun);
            let c = select(vec3(1.0, 1.0 - min(-nd * 8.0, 1.0), 1.0 - min(-nd * 8.0, 1.0)),
                           vec3(1.0 - min(nd * 8.0, 1.0), 1.0, 1.0 - min(nd * 8.0, 1.0)), nd >= 0.0);
            return vec4(c * (0.4 + 0.6 * lit), 1.0);
        }
        // 6: each cascade tinted (red, green, blue, yellow), to see where they reach.
        if (debug_view > 5.5 && debug_view < 6.5) {
            var tint = array<vec3<f32>, 4>(vec3(1.0, 0.6, 0.6), vec3(0.6, 1.0, 0.6), vec3(0.6, 0.6, 1.0), vec3(1.0, 1.0, 0.5))[k];
            return vec4(tint * (0.4 + 0.6 * lit), 1.0);
        }
    } else if (debug_view > 0.5 && debug_view < 7.5) {
        return vec4(vec3(0.5), 1.0);
    }
    // Wind Waker HD's facing term (g_sunFacing): what faces away from the sun is in shadow too, from the
    // normal rebuilt from depth, everywhere up to HD's far fade (in no cascade as well); not on characters
    // (the mask's green), which keep their own shading, nor on the sea. Then HD's terrain curve.
    // Wind Waker HD's terrain mix (its shadow pre-pass and terrain shaders): the quarter-size pass writes the
    // map's shadow and the grazing term 10 x (1 - |N.L|)^9; the main pass takes the blurred shadow by
    // distance, grazing and nearness to the camera, then HD's curve 0.5 x (2s)^e, e from 4 to 2 by distance
    // and to 1 where the blurred shadow is taken (a soft edge stays half lit). Not on the sea or characters.
    if (low_pass > 0.5) {
        let graze = 10.0 * pow(clamp(1.0 - abs(dot(normal, sun)), 0.0, 1.0), 9.0);
        // HD's SSAO in blue, blurred with the rest (HD's shadow pre-pass keeps it beside its shadow).
        let ao = select(1.0, hd_ao(in.uv, z, in.pos.xy), view.facing.w > 0.5);
        // Fitted cascades: lit toward their far end, as HD's pre-pass's far fade (it adds, its target clamps).
        if (hd_prepass > 0.5 && view.fit.x > 0.5) {
            lit = min(lit + clamp((z - view.fit.z) / max(view.fit.w - view.fit.z, 1.0), 0.0, 1.0), 1.0);
        }
        return vec4(lit, graze, ao, 1.0);
    }
    // What is the sea's surface for the light's terms below: the sea mask's red where the frame has it; else what
    // lies at the sea's height. By height alone a cliff's foot (its lowest 40 to 80 units) counted as sea, took no
    // facing term or terrain mix and stood lit in a band along the water (Outset, left of the bridge, 10:00).
    var water = sea;
    if (view.facing.x > 0.5) {
        let wsize = vec2<f32>(textureDimensions(mask));
        let wp = clamp(vec2<i32>(in.uv * wsize), vec2<i32>(0), vec2<i32>(wsize) - vec2<i32>(1));
        water = textureLoad(mask, wp, 0).r;
    }
    // HD's terrain mix's fades: with the cascades fitted to its live splits, HD's own rule (its terrain constants
    // at Outset's bridge at 10:00 and 16:00): the curve from the first split over 200 units, the blur over the 0.4
    // of the way between the splits ending at the second, so the far switch is always under the blur.
    var e_start = mix_e_start;
    var blur_start = mix_blur_start;
    var blur_end = mix_blur_end;
    if (view.fit.x > 0.5) {
        e_start = view.hd.x;
        blur_end = view.hd.y;
        blur_start = view.hd.y - 0.4 * (view.hd.y - view.hd.x);
    }
    // HD's pass: every pixel's shadow from the blurred pre-pass (bilinear, as HD's materials read it). The sea
    // takes its mip 1 by distance as the terrain does (below): far off, a palm's shadow on the water is soft.
    if (hd_prepass > 0.5) {
        lit = textureSampleLevel(shadow_blur, samp, in.uv, 0.0).r;
        if (water > 0.5) {
            let fsea = clamp((z - blur_start) / max(blur_end - blur_start, 1.0), 0.0, 1.0);
            lit = mix(lit, textureSampleLevel(shadow_blur, samp, in.uv, 1.0).r, fsea);
        }
    }
    // HD's SSAO on the scenery: on its ambient alone (the shade), faded on faces square to the camera.
    var ao_term = 1.0;
    if (hd_mix > 0.5 && water < 0.5) {
        var chr = false;
        if (view.facing.x > 0.5) {
            let csize = vec2<f32>(textureDimensions(mask));
            let cp = clamp(vec2<i32>(in.uv * csize), vec2<i32>(0), vec2<i32>(csize) - vec2<i32>(1));
            chr = textureLoad(mask, cp, 0).g > character_g;
        }
        if (!chr) {
            var b = textureSampleLevel(shadow_blur, samp, in.uv, 0.0);
            if (hd_prepass > 0.5) {
                // HD's mip 1: half the pre-pass's size, blurred again.
                b.r = textureSampleLevel(shadow_blur, samp, in.uv, 1.0).r;
            }
            let near = 1.0 - clamp((z - 60.0) * 0.1, 0.0, 1.0);
            let fblur = clamp((z - blur_start) / max(blur_end - blur_start, 1.0), 0.0, 1.0);
            let w = clamp(fblur + b.g + near, 0.0, 1.0);
            let s = mix(lit, b.r, w);
            let e = mix(mix(mix_e_near, 2.0, clamp((z - e_start) * 0.005, 0.0, 1.0)), 1.0, w);
            lit = clamp(0.5 * pow(2.0 * s, e), 0.0, 1.0);
            if (view.facing.w > 0.5) {
                let nz = clamp(depth_normal(in.uv, p, tx, ty).z, 0.0, 1.0);
                ao_term = clamp(pow(b.b, ao_power) + select(0.0, pow(nz, 20.0), ao_facing > 0.5), 0.0, 1.0);
            }
        }
    }
    var facing = 1.0;
    var terrain = false;
    let mapLit = lit;
    if (facing_on > 0.5 && view.facing.x > 0.5) {
        let msize = vec2<f32>(textureDimensions(mask));
        let mp = clamp(vec2<i32>(in.uv * msize), vec2<i32>(0), vec2<i32>(msize) - vec2<i32>(1));
        let mk = textureLoad(mask, mp, 0);
        terrain = mk.r < 0.5 && mk.g < character_g;
        if (terrain) {
            // The scenery's smooth normal where its draw gave one (blended across each triangle, as HD's
            // models' per-vertex light is), else the depth's (one per triangle).
            let nd = depth_normal(in.uv, p, tx, ty);
            var nv = nd;
            var smoothed = false;
            if (view.facing.z > 0.5) {
                let sn = textureLoad(normals, mp, 0);
                if (sn.a > 0.5) {
                    nv = normalize(sn.xyz * 2.0 - 1.0);
                    smoothed = true;
                }
            }
            let nw = vec3(dot(view.rows[0].xyz, nv), dot(view.rows[1].xyz, nv), dot(view.rows[2].xyz, nv));
            let away = 1.0 - smoothstep(facing_far_start, facing_far_end, z);
            var turned = clamp(dot(nw, sun) * facing_mul, 0.0, 1.0);
            // The shape's own facing too where the light came from smooth normals (HD's shadow pre-pass).
            if (smoothed && facing_depth_mul > 0.0) {
                if (facing_depth_blur > 0.0) {
                    turned = min(turned, depth_facing_blurred(in.uv, z, tx, ty, sun));
                } else {
                    let ndw = vec3(dot(view.rows[0].xyz, nd), dot(view.rows[1].xyz, nd), dot(view.rows[2].xyz, nd));
                    turned = min(turned, clamp(dot(ndw, sun) * facing_depth_mul, 0.0, 1.0));
                }
            }
            // Skinned scenery without normals of its own (the mask's green 0.6: a palm's leaves, whose draws carry
            // none) has no facing term: the depth's normal on a cut-out leaf card faces anywhere, and it left the
            // crowns in shade (Outset, 10:00). Wind Waker HD lights them by its own model's normals, which this
            // doesn't have; the shadow map alone shades their undersides.
            if (!smoothed && mk.g > 0.3) {
                turned = 1.0;
            }
            // With smooth normals made for it, a leaf is lit from either side: the mask's pass turns the normal to
            // the side seen, and from below that is away from the sun, which left the crowns dark. HD's leaf normals
            // point off the frond's top whichever side is seen.
            if (smoothed && mk.g > 0.3) {
                turned = clamp(abs(dot(nw, sun)) * facing_mul, 0.0, 1.0);
            }
            facing = mix(1.0, turned, away);
            // HD multiplies its light amount by the shadow (R2.w x shade); BlueWake's own takes the lesser.
            lit = select(min(lit, facing), lit * facing, hd_mix > 0.5);
            if (shape_on > 0.5) {
                let e = mix(4.0, 2.0, smoothstep(shape_near, shape_far, z));
                lit = clamp(0.5 * pow(2.0 * lit, e), 0.0, 1.0);
            }
        }
    }
    // 12: with DOL_AURORA_SUN_SHADOW_TEMPORAL_DEBUG=1, how much the pre-pass's shadow changed since the last
    // frame at the same place in the world (black: from lit to full shadow or back).
    if (debug_view > 11.5) {
        return vec4(vec3(textureSampleLevel(shadow_blur, samp, in.uv, 0.0).g), 1.0);
    }
    // 10: the facing term alone (white facing the sun, black turned away; green where it came from the
    // scenery's smooth normals). 11: the shadow map alone, before the facing term.
    if (debug_view > 10.5) {
        return vec4(vec3(mapLit), 1.0);
    }
    if (debug_view > 9.5) {
        var fromNormals = false;
        if (view.facing.z > 0.5) {
            let msize = vec2<f32>(textureDimensions(mask));
            let mp = clamp(vec2<i32>(in.uv * msize), vec2<i32>(0), vec2<i32>(msize) - vec2<i32>(1));
            fromNormals = textureLoad(normals, mp, 0).a > 0.5;
        }
        return vec4(select(vec3(facing), vec3(0.6, 1.0, 0.6) * facing, fromNormals && terrain), 1.0);
    }
    // 9: how lit each pixel comes out, all terms together (white lit, black shade), to set against Wind
    // Waker HD's own shadow; what the facing term leaves out (characters, the sea) tinted blue, and the
    // whole view tinted pink when the facing term is off or the frame has no sea mask.
    if (debug_view > 8.5) {
        var tint = vec3(1.0);
        if (facing_on < 0.5 || view.facing.x < 0.5) {
            tint = vec3(1.0, 0.75, 0.75);
        } else if (!terrain) {
            tint = vec3(0.6, 0.7, 1.0);
        }
        return vec4(tint * (0.1 + 0.9 * lit), 1.0);
    }
    // 8: the scene as it is, with the map's shadow in pink (its green taken
    // out by how shaded it is), to find shadows on what shouldn't have any.
    if (debug_view > 7.5) {
        return vec4(1.0, lit, 1.0, 1.0);
    }
    if (debug_view > 0.5) {
        return vec4(vec3(lit), 1.0);
    }
    // The screen's steps, a short way (fs_sun), when asked for: on their own
    // they also found the thin parts of what the map holds (hair, an arm) and
    // drew dotted rays behind Link.
    if (screen_steps > 0.5) {
    let l = normalize(vec3(dot(vec3(view.rows[0].x, view.rows[1].x, view.rows[2].x), sun),
                           dot(vec3(view.rows[0].y, view.rows[1].y, view.rows[2].y), sun),
                           dot(vec3(view.rows[0].z, view.rows[1].z, view.rows[2].z), sun)));
    let jitter = fract(sin(dot(in.pos.xy, vec2(12.9898, 78.233))) * 43758.5453);
    for (var i = 0; i < 8; i++) {
        let t = 10.0 + (contact - 10.0) * (f32(i) + jitter) / 8.0;
        let q = p + l * t;
        if (-q.z <= view.lens.z) {
            break;
        }
        let s = vec2(q.x / -q.z / tx, q.y / -q.z / ty);
        let suv = vec2(s.x * 0.5 + 0.5, 0.5 - s.y * 0.5);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
            break;
        }
        let surface = view_depth(suv);
        let behind = -q.z - surface;
        if (surface > 0.0 && behind > 12.0 + 0.004 * surface && behind < 250.0 && !in_sun_map(s, surface, tx, ty)) {
            lit = min(lit, smoothstep(1500.0, 4000.0, z));  // near the camera only
            break;
        }
    }
    }
    // The cloud shadows on Wind Waker HD's rule (g_cloudHd): its shadow term times clamp(cloud + floor), the floor
    // 1 - max(0.25, -0.9 cos(the hour's sun angle)) (cCloudShadowPower, cloud_hd_floor()), so they darken only what the sun
    // lights, and are lighter at a low sun (0.75 at night). The cloud is BlueWake's own drifting noise (kCloudNoise), 1 clear, 0 under a cloud.
    if (view.shade.w > 0.5) {
        if (view.lit.w > 0.5) {
            // HD's: its texture from above, its mip from how much ground a pixel spans (after an early return, so
            // no implicit derivatives here).
            let uv = vec2(-world.x / cloud_repeat + view.time.x * cloud_speed, -world.z / cloud_repeat + 0.5);
            let span = 2.0 * view.lens.x * z / f32(textureDimensions(depth).y);
            let lod = log2(max(span * f32(textureDimensions(cloud_map).x) / cloud_repeat, 1.0));
            lit *= clamp(textureSampleLevel(cloud_map, cloud_samp, uv, lod).r + view.palette.w, 0.0, 1.0);
        } else {
            let drift = vec2(0.6, 0.25) * view.time.x * 0.02;
            let cloud = 1.0 - smoothstep(0.42, 0.6, fbm(world.xz / cloud_scale + drift));
            lit *= mix(clamp(cloud + view.palette.w, 0.0, 1.0), 1.0, smoothstep(60000.0, 160000.0, z));
        }
    }
    if (ao_debug > 0.5) {
        return vec4(vec3(ao_term), 1.0);
    }
    // Wind Waker HD's character light (view.hd.z) has the shadow in the characters' own draws: those drawn with it
    // (the mask's alpha 1) are left as drawn.
    if (view.hd.z > 0.5 && view.facing.x > 0.5) {
        let tsize = vec2<f32>(textureDimensions(mask));
        let tp = clamp(vec2<i32>(in.uv * tsize), vec2<i32>(0), vec2<i32>(tsize) - vec2<i32>(1));
        if (textureLoad(mask, tp, 0).a > 0.5) {
            return vec4(1.0);
        }
    }
    // An effect drawn over the pixel (the spin attack's rings, a glow) isn't in
    // the shade of the scenery behind it, whose depth this is: Link's shadow
    // was drawn across the rings. As much of the pixel as they make up is left.
    let cover = effect_cover(in.uv, z);
    if (effect_debug > 0.5) {
        return vec4(1.0, 1.0 - cover, 1.0 - cover, 1.0);
    }
    let fade = view.params.z * (1.0 - cover);
    // HD's SSAO darkens its ambient: mix(ambient x AO, sunlight, lit), against the sunlight.
    var shade = mix(vec3(1.0), view.shade.rgb * ao_term, (1.0 - lit) * fade);
    var shade_lin = vec3(1.0);  // with HD's palette: the multiplier in linear light
    if (view.facing.y > 0.5 && fade * (1.0 - lit) > 0.0) {
        // Wind Waker HD's light (its terrain shaders, read in Cemu): its shade takes the sky's light from above
        // and the ambient from the side (0.5 + 0.5 N.y between them), against its sunlight; that ratio, in the
        // scene's gamma, is what a shaded pixel is multiplied by.
        var nv = depth_normal(in.uv, p, tx, ty);
        var up = dot(vec3(view.rows[1].x, view.rows[1].y, view.rows[1].z), nv);
        // Scenery with smooth normals: by them, as HD's per-vertex light (its terrain VS takes the hemisphere from
        // its own vertex normals, blended across each triangle), not by the depth's face. HD's ground normals lean
        // where the ground meets a slope or a wall, so its shade there takes some of the ambient from the side (at
        // Windfall's spawn 7 path 0.5 + 0.5 N.y is 0.94-0.96, not 1); by the depth's face the ground took the sky's
        // colour alone, bluer than HD's. Skinned scenery (a palm): either side of a leaf as up as the side seen (the
        // mask's pass turned the normal to it; HD's leaf normals point off the top). smooth_ambient 2: skinned only.
        if (smooth_ambient > 0.5 && view.facing.x > 0.5 && view.facing.z > 0.5) {
            let asize = vec2<f32>(textureDimensions(mask));
            let ap = clamp(vec2<i32>(in.uv * asize), vec2<i32>(0), vec2<i32>(asize) - vec2<i32>(1));
            let sn = textureLoad(normals, ap, 0);
            let am = textureLoad(mask, ap, 0);
            if (sn.a > 0.5 && am.r < 0.5 && am.g < character_g) {
                let row = vec3(view.rows[1].x, view.rows[1].y, view.rows[1].z);
                let snv = normalize(sn.xyz * 2.0 - 1.0);
                if (am.g > 0.3) {
                    nv = snv;
                    up = abs(dot(row, nv));
                } else if (smooth_ambient < 1.5) {
                    // HD's hemisphere is per vertex (0.5 + 0.5 N.y at each, blended linearly), so the blended
                    // normal's y as written, not renormalized: renormalizing leans it up between vertices whose
                    // normals differ (Windfall's spawn 7 path: (-0.93, 0, -0.37) to (-0.60, 0.80, 0.05)).
                    nv = snv;
                    up = dot(row, sn.xyz * 2.0 - 1.0);
                }
            }
        }
        let ambient = mix(view.palette.rgb, view.shade.rgb, 0.5 + 0.5 * up);
        let ratio = clamp(ambient / max(view.lit.rgb, vec3(1e-3)), vec3(0.0), vec3(1.0));
        shade_lin = mix(vec3(1.0), ratio * ao_term, (1.0 - lit) * fade);
        shade = pow(shade_lin, vec3(1.0 / 2.2));
    }
    // Wind Waker HD's light buffer on the lit scenery (its draws added 1.25 x P to their light, view.points.z): HD
    // adds it after the shade, times its AO alone, (vtx x L + 1.25 x AO x P) x albedo. The scene holds the sunlit
    // light plus the lamps' (taken as HD's sunlight, view.lit, plus 1.25 x P here), so its multiplier is
    // (lit x shade + 1.25 x P x AO) / (lit + 1.25 x P): a lamp's light on shaded scenery isn't darkened with the
    // sunlight, and takes HD's AO where the sun is. Not on the sea or characters, which take the lamps their own way.
    if (view.points.z > 0.5 && view.facing.y > 0.5 && view.points.x > 0.5 && water < 0.5) {
        var chr = false;
        if (view.facing.x > 0.5) {
            let csize = vec2<f32>(textureDimensions(mask));
            let cp = clamp(vec2<i32>(in.uv * csize), vec2<i32>(0), vec2<i32>(csize) - vec2<i32>(1));
            chr = textureLoad(mask, cp, 0).g > character_g;
        }
        if (!chr) {
            var nv = depth_normal(in.uv, p, tx, ty);
            if (view.facing.z > 0.5) {
                let nsize = vec2<f32>(textureDimensions(normals));
                let np = clamp(vec2<i32>(in.uv * nsize), vec2<i32>(0), vec2<i32>(nsize) - vec2<i32>(1));
                let sn = textureLoad(normals, np, 0);
                if (sn.a > 0.5) {
                    nv = normalize(sn.xyz * 2.0 - 1.0);
                }
            }
            let nw = vec3(dot(view.rows[0].xyz, nv), dot(view.rows[1].xyz, nv), dot(view.rows[2].xyz, nv));
            let q = view.points.y * point_sum(world.xyz, nw);
            if (max(q.r, max(q.g, q.b)) > 1e-4) {
                let k = max(view.lit.rgb, vec3(0.02));
                shade = pow((k * shade_lin + q * ao_term) / (k + q), vec3(1.0 / 2.2));
            }
        }
    }
    // Wind Waker HD's lava glow (gxcore_draw.cpp hd_lava_draw()): HD adds it outside the shade, so its share of the
    // scenery's lit light, 1 - the normals' alpha over 0.49 where the sea mask's pass put it there, is left unshaded.
    if (view.facing.z > 0.5 && view.facing.y > 0.5 && water < 0.5) {
        let gsize = vec2<f32>(textureDimensions(normals));
        let gp = clamp(vec2<i32>(in.uv * gsize), vec2<i32>(0), vec2<i32>(gsize) - vec2<i32>(1));
        let ga = textureLoad(normals, gp, 0).a;
        if (ga > 0.5 && ga < 0.995) {
            let glow = clamp((1.0 - ga) / 0.49, 0.0, 1.0);
            shade = pow(mix(pow(shade, vec3(2.2)), vec3(1.0), glow), vec3(1.0 / 2.2));
        }
    }
    // The sea's shadow at sea_shadow of its strength (HD's cPrimShadowIntensity, 0.5 in its pass), where the sea mask
    // has the sea.
    if (sea_shadow < 0.999 && view.facing.x > 0.5) {
        let ssize = vec2<f32>(textureDimensions(mask));
        let sp = clamp(vec2<i32>(in.uv * ssize), vec2<i32>(0), vec2<i32>(ssize) - vec2<i32>(1));
        // With HD's light (view.facing.y), HD's own sea: base x (1 - s + s x clamp(cloud + floor) x shadow) in
        // linear light (its sea quad and wave grid, PS 949 and 951, s = 0.5): lit here already holds the shadow
        // times the cloud term, and the sea takes neither the shade's ambient over the sunlight nor the AO.
        var sea_shade = mix(vec3(1.0), shade, sea_shadow);
        if (view.facing.y > 0.5) {
            sea_shade = vec3(pow(max(1.0 - sea_shadow * (1.0 - lit) * fade, 0.0), 1.0 / 2.2));
        }
        shade = mix(shade, sea_shade, textureLoad(mask, sp, 0).r);
    }
    // Multiplied into the 8-bit scene: half a step of noise against bands (fs_ao_apply).
    let noise = (fract(52.9829189 * fract(dot(in.pos.xy, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
    return vec4(shade + select(0.0, noise, direct > 0.5 && shade.r < 0.999), 1.0);
}

@fragment
fn fs_sun_map(in: VertexOutput) -> @location(0) vec4<f32> {
    return sun_map_shade(in);
}
)""";
// With dual-source blending (the scene times the second output, plus the first): the shade, and the lit scenery's
// fog after it, as Wind Waker HD fogs its scenery (its room and terrain shaders: mix(lit colour, fog, f), f the
// square of (the distance from the eye - start) / (end - start)). The draws fogged it before the shade, by the
// depth: the pass takes their fog out and puts HD's in, (scene - F fs) / (1 - fs) x shade x (1 - fh) + F fh, in the
// scene's gamma, as GX fogs. Without it a shaded wall's fog was shaded too, and the far walls of a dungeon room
// (most of them fog in HD) came out at under half HD's.
constexpr const char* kSunMapFogEntry = R"""(
struct SunFogOut {
    @location(0) @blend_src(0) add: vec4<f32>,
    @location(0) @blend_src(1) mul: vec4<f32>,
};

@fragment
fn fs_sun_map_fog(in: VertexOutput) -> SunFogOut {
    let m = sun_map_shade(in);
    var out: SunFogOut;
    out.add = vec4(0.0);
    out.mul = m;
    if (view.fog_colour.w < 0.5 || view.facing.y < 0.5 || debug_view > 0.5 || effect_debug > 0.5 || ao_debug > 0.5) {
        return out;
    }
    let p = vec2<i32>(in.pos.xy);
    let d = textureLoad(depth, p, 0);
    if (d <= 0.0) {
        return out;
    }
    // Not the sea (its own fog) or the characters (fogged in their own draws).
    if (view.facing.x > 0.5) {
        let msize = vec2<f32>(textureDimensions(mask));
        let mp = clamp(vec2<i32>(in.uv * msize), vec2<i32>(0), vec2<i32>(msize) - vec2<i32>(1));
        let mk = textureLoad(mask, mp, 0);
        if (mk.r > 0.5 || mk.g > character_g) {
            return out;
        }
    }
    // The draws' fog (GX's perspective fog, as the scene's draws took it).
    let zs = clamp(i32((1.0 - d) * 16777216.0), 0, 16777215);
    let ze = view.fog.x * 16777216.0 / (view.fog.y - f32(zs >> u32(view.fog.z)));
    // GX's range adjust, as the draws took it (across the screen only).
    var adjust = 1.0;
    if (view.fog_x.x > 0.5) {
        let offset = (2.0 * (in.pos.x / view.fog_x.z)) - 1.0 - view.fog_x.y;
        let index = clamp(9.0 - abs(offset) * 9.0, 0.0, 9.0);
        let lower = u32(index);
        let upper = min(lower + 1u, 9u);
        let k = mix(view.fog_range[lower >> 2u][lower & 3u], view.fog_range[upper >> 2u][upper & 3u], fract(index));
        adjust = sqrt(offset * offset + k * k) / max(k, 1e-6);
    }
    var fs = clamp(ze * adjust - view.fog.w, 0.0, 1.0);
    // HD's: the same distances, from the eye rather than along the view.
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let ray = length(vec3(ndc.x * view.lens.x * view.lens.y, ndc.y * view.lens.x, 1.0));
    var fh = clamp(ze * ray - view.fog.w, 0.0, 1.0);
    // A start the game clamped at 0 (C 0): its end is the view depth over ze; where an HD entry has that end and a
    // start below 0 (aurora_set_fog_hd_starts), HD's fog from that start.
    if (abs(view.fog.w) < 1e-6 && ze > 1e-9) {
        let vz = view_depth(in.uv);
        let end = vz / ze;
        for (var i = 0u; i < 8u; i++) {
            let pair = select(view.fog_starts[i >> 1u].zw, view.fog_starts[i >> 1u].xy, (i & 1u) == 0u);
            if (pair.y > 0.0 && abs(end - pair.y) < 0.01 * pair.y) {
                fh = clamp((vz * ray - pair.x) / (pair.y - pair.x), 0.0, 1.0);
                break;
            }
        }
    }
    if (view.fog_colour.w > 1.5) {
        fs = fs * fs;
        fh = fh * fh;
    }
    let keep = clamp(m.rgb * (1.0 - fh) / max(1.0 - fs, 1.0 / 256.0), vec3(0.0), vec3(1.0));
    out.mul = vec4(keep, 1.0);
    out.add = vec4(max(view.fog_colour.rgb * (fh - fs * keep), vec3(0.0)), 0.0);
    return out;
}
)""";
wgpu::ShaderModule g_sunMapModule;
wgpu::BindGroupLayout g_sunMapLayout;
wgpu::PipelineLayout g_sunMapPipelineLayout;
wgpu::RenderPipeline g_sunMapPass;
wgpu::RenderPipeline g_sunMapDirect; // multiplied straight into the scene
wgpu::Sampler g_sunMapCompare;
wgpu::Texture g_sunMapTex;
wgpu::TextureView g_sunMapView;
std::array<wgpu::TextureView, kSunMapCascades> g_sunMapLayers;

// Edge light: a warm rim on the sunlit outline of what is near the camera (Link,
// people, objects). A pixel is on such an outline when, a few pixels from it
// toward the sun on screen, the depth jumps far behind it or there is none.
constexpr const char* kEdgeShader = R"""(
override strength: f32 = 0.4;

struct View {
    rows: array<vec4<f32>, 3>,
    lens: vec4<f32>,
    time: vec4<f32>,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var unused: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;

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

fn view_depth(p: vec2<i32>) -> f32 {
    let size = vec2<i32>(textureDimensions(depth));
    let d = textureLoad(depth, clamp(p, vec2<i32>(0), size - vec2<i32>(1)), 0);
    return select(view.lens.z / d, 0.0, d <= 0.0);
}

@fragment
fn fs_edge(in: VertexOutput) -> @location(0) vec4<f32> {
    let here = vec2<i32>(in.pos.xy);
    let z = view_depth(here);
    let sun = view.time.yzw;
    if (z <= 0.0 || z > 1500.0 || sun.y < 0.05) {
        return vec4(0.0);
    }
    // The sun's direction in view space, then on screen (y down).
    let l = vec3(dot(vec3(view.rows[0].x, view.rows[1].x, view.rows[2].x), sun),
                 dot(vec3(view.rows[0].y, view.rows[1].y, view.rows[2].y), sun),
                 dot(vec3(view.rows[0].z, view.rows[1].z, view.rows[2].z), sun));
    var dir = vec2(l.x, -l.y);
    if (length(dir) < 0.05) {
        dir = vec2(0.0, -1.0);
    }
    dir = normalize(dir);
    let scale = f32(textureDimensions(depth).y) / 1080.0;
    var rim = 0.0;
    for (var i = 1; i <= 3; i++) {
        let n = view_depth(here + vec2<i32>(round(dir * f32(i) * 1.5 * scale)));
        if (n <= 0.0 || n > z * 1.5 + 100.0) {
            rim = max(rim, 1.0 - f32(i - 1) / 3.0);
        }
    }
    let fade = 1.0 - smoothstep(800.0, 1500.0, z);
    return vec4(vec3(1.0, 0.95, 0.82) * rim * strength * fade, 0.0);
}
)""";
wgpu::ShaderModule g_edgeModule;

// Torch and fire light: the game's own point lights (torches, fires, lanterns,
// bombs) glow on the surfaces around them, each pixel's place in the world from
// the camera and the depth, with a smooth falloff to the light's radius.
constexpr const char* kTorchShader = R"""(
override strength: f32 = 0.35;
override radius_scale: f32 = 3.0;

struct Light {
    place: vec4<f32>,  // x, y, z, radius
    color: vec4<f32>,  // r, g, b (0 to 1), flicker
};

struct View {
    rows: array<vec4<f32>, 3>,
    lens: vec4<f32>,
    time: vec4<f32>,
    count: vec4<f32>,
    lights: array<Light, 32>,
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var unused: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;

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
fn fs_torch(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = textureLoad(depth, vec2<i32>(in.pos.xy), 0);
    if (d <= 0.0) {
        return vec4(0.0);
    }
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let ray = vec3(ndc.x * view.lens.x * view.lens.y, ndc.y * view.lens.x, -1.0);
    let dir = vec3(dot(view.rows[0].xyz, ray), dot(view.rows[1].xyz, ray), dot(view.rows[2].xyz, ray));
    let eye = vec3(view.rows[0].w, view.rows[1].w, view.rows[2].w);
    let world = eye + dir * (view.lens.z / d);
    var glow = vec3(0.0);
    let n = u32(view.count.x);
    for (var i = 0u; i < 32u; i++) {
        if (i >= n) {
            break;
        }
        let light = view.lights[i];
        let falloff = clamp(1.0 - length(world - light.place.xyz) / (light.place.w * radius_scale), 0.0, 1.0);
        let flicker = 1.0 + light.color.w * 0.15 * sin(view.time.x * 9.0 + f32(i) * 1.7);
        glow += light.color.rgb * falloff * falloff * flicker;
    }
    return vec4(glow * strength, 0.0);
}
)""";
wgpu::ShaderModule g_torchModule;
wgpu::BindGroupLayout g_torchLayout;
wgpu::PipelineLayout g_torchPipelineLayout;
wgpu::RenderPipeline g_torchPass;
uint32_t g_torchSamples = 0;
constexpr size_t kTorchUniformSize = 20 * sizeof(float) + 4 * sizeof(float) + kMaxLights * 8 * sizeof(float);
wgpu::RenderPipeline g_edgePass;
uint32_t g_edgeSamples = 0;

// DOL_AURORA_RIM_HD=1 (local test; in place of the edge light above): Wind Waker HD's rim ("edge") light on
// characters, as its toon materials add it (wii_pipeline: ambient x colour x (1 + N.V)^(sharpness x 10) x
// intensity, more against the light), with HD's defaults for every character (colour 1, intensity 2, sharpness
// 0.5, rim 0.75; 1,509 of its 1,763 rim materials). A first cut as a pass over the scene: the characters are the
// sea mask's green, their normals their own from the sea mask's pass (the depth's where a part has none), and the
// ambient HD's hemisphere from its palette over its sunlight, added on top (HD adds it to the texture before its
// light, so this misses the material's colour and shading).
bool g_rimHd = false;
float g_rimHdScale = 1.f;     // DOL_AURORA_RIM_HD_SCALE: the rim times this
float g_rimHdIntensity = 2.f; // DOL_AURORA_RIM_HD_INTENSITY, _SHARPNESS, _RIM: HD's defaults
float g_rimHdSharpness = 0.5f;
float g_rimHdRim = 0.75f;
int g_rimHdDebug = 0;         // DOL_AURORA_RIM_HD_DEBUG=1: the rim alone, over black
constexpr const char* kRimShader = R"""(
override strength: f32 = 1.0;
override intensity: f32 = 2.0;
override sharpness: f32 = 0.5;
override rim: f32 = 0.75;

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds, then the direction to the sun (world)
    sky: vec4<f32>,             // HD's palette (linear): its light from above; w: 1 with the palette
    ground: vec4<f32>,          // from the side; w: 1 when the characters took Wind Waker HD's light (actor_light_applied())
    lit: vec4<f32>,             // its sunlight
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var mask: texture_2d<f32>;    // the sea mask: green a character
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var normals: texture_2d<f32>; // view-space normals x 0.5 + 0.5, alpha 1 where a draw gave one

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
fn fs_rim(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy);
    let d = textureLoad(depth, p, 0);
    let z = select(view.lens.z / d, 0.0, d <= 0.0);
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let v = vec3(ndc.x * view.lens.x * view.lens.y, ndc.y * view.lens.x, -1.0) * z;
    // The depth's normal (before any return: derivatives need every pixel), turned to the camera.
    var nd = cross(dpdx(v), dpdy(v));
    nd = select(vec3(0.0, 0.0, 1.0), normalize(nd), length(nd) > 1e-12);
    nd = select(nd, -nd, dot(nd, -v) < 0.0);
    // Toon-shaded characters only (green 1; a skinned palm's leaves are 0.6), and not those drawn with Wind Waker
    // HD's light (the mask's alpha 1, while ground.w says it was applied), which has the rim in it.
    let mk = textureLoad(mask, p, 0);
    if (d <= 0.0 || mk.g < 0.8 || (view.ground.w > 0.5 && mk.a > 0.5)) {
        return vec4(0.0);
    }
    let nt = textureLoad(normals, p, 0);
    let n = select(nd, normalize(nt.xyz * 2.0 - 1.0), nt.a > 0.5);
    let eye_dir = normalize(v);
    // 0 facing the camera, 1 at the silhouette.
    let e = clamp(1.0 + dot(n, eye_dir), 0.0, 1.0);
    // HD's reverse light: 1 looking toward the key light (backlit), 1 - rim with it behind the camera.
    let sun = view.time.yzw;
    let sun_view = normalize(vec3(dot(vec3(view.rows[0].x, view.rows[1].x, view.rows[2].x), sun),
                                  dot(vec3(view.rows[0].y, view.rows[1].y, view.rows[2].y), sun),
                                  dot(vec3(view.rows[0].z, view.rows[1].z, view.rows[2].z), sun)));
    let reverse = mix(1.0, clamp(dot(eye_dir, sun_view), 0.0, 1.0), rim);
    // HD's hemisphere ambient (from the side below, from above at the top), over its sunlight, in the scene's gamma.
    let up = dot(view.rows[1].xyz, n);
    var amb = vec3(0.6);
    if (view.sky.w > 0.5) {
        amb = clamp(mix(view.ground.rgb, view.sky.rgb, 0.5 + 0.5 * up) / max(view.lit.rgb, vec3(1e-3)), vec3(0.0), vec3(1.0));
        amb = pow(amb, vec3(1.0 / 2.2));
    }
    return vec4(amb * pow(e, sharpness * 10.0) * intensity * reverse * strength, 0.0);
}
)""";
wgpu::ShaderModule g_rimModule;
wgpu::RenderPipeline g_rimPass;
uint32_t g_rimSamples = 0;
wgpu::RenderPipeline g_sunPass;
uint32_t g_sunSamples = 0;
wgpu::BindGroupLayout g_cloudLayout;
wgpu::PipelineLayout g_cloudPipelineLayout;
wgpu::RenderPipeline g_cloudPass;
uint32_t g_cloudSamples = 0;

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

enum Blend { kReplace, kScreen, kMultiply };

wgpu::RenderPipeline make_pipeline(const char* label, const char* entry, wgpu::TextureFormat format,
                                   uint32_t samples, Blend mode) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "threshold", .value = g_threshold},
      wgpu::ConstantEntry{.key = "strength", .value = g_strength},
  };
  // Screen: the glow brightens what is darker and tends to white instead of
  // passing it. Added, it clipped Wind Waker's near-white clouds to flat white.
  const wgpu::BlendState screen{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::OneMinusDst,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const wgpu::BlendState multiply{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{
      .format = format,
      .blend = mode == kScreen ? &screen : mode == kMultiply ? &multiply : nullptr,
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

wgpu::RenderPipeline make_ao_pipeline(const char* label, const char* entry, wgpu::TextureFormat format,
                                      uint32_t samples, Blend mode) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "radius", .value = g_aoRadius},
      wgpu::ConstantEntry{.key = "strength", .value = g_aoStrength},
      wgpu::ConstantEntry{.key = "fade_start", .value = g_aoFadeStart},
      wgpu::ConstantEntry{.key = "fade_end", .value = g_aoFadeEnd},
      wgpu::ConstantEntry{.key = "debug_view", .value = static_cast<double>(g_aoDebug)},
      wgpu::ConstantEntry{.key = "effect_gain", .value = g_effectGain},
  };
  const wgpu::BlendState multiply{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{
      .format = format, .blend = mode == kMultiply ? &multiply : nullptr, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_aoModule,
      .entryPoint = entry,
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      .layout = g_depthPipelineLayout,
      .vertex = {.module = g_aoModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_cloud_pipeline(wgpu::TextureFormat format, uint32_t samples, bool exposure = false) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_cloudStrength},
      wgpu::ConstantEntry{.key = "scale", .value = g_cloudScale},
      wgpu::ConstantEntry{.key = "debug_view", .value = g_cloudDebug ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "effect_gain", .value = g_effectGain},
  };
  const wgpu::BlendState boost{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Dst,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const wgpu::BlendState multiply{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{
      wgpu::ColorTargetState{.format = format, .blend = exposure ? &boost : &multiply, .writeMask = wgpu::ColorWriteMask::All}};
  static const bool seaDebug = std::getenv("DOL_AURORA_SEA_MASK_DEBUG") != nullptr;
  static const std::array<float, 2> exposureFar = [] {
    std::array<float, 2> far{10000.f, 25000.f};
    if (const char* value = std::getenv("DOL_AURORA_EXPOSURE_FAR"); value != nullptr)
      if (std::sscanf(value, "%f,%f", &far[0], &far[1]) != 2 || !(far[1] > far[0]) || !(far[0] >= 0.f))
        far = {10000.f, 25000.f};
    return far;
  }();
  static const std::array<float, 2> exposureHighlights = [] {
    std::array<float, 2> range{0.4f, 1.0f};
    if (const char* value = std::getenv("DOL_AURORA_EXPOSURE_HIGHLIGHTS"); value != nullptr)
      if (std::sscanf(value, "%f,%f", &range[0], &range[1]) != 2 || !(range[1] > range[0]))
        range = {0.4f, 1.0f};
    return range;
  }();
  const std::array exposureConstants{wgpu::ConstantEntry{.key = "sea_debug", .value = seaDebug ? 1.0 : 0.0},
                                     wgpu::ConstantEntry{.key = "far_start", .value = exposureFar[0]},
                                     wgpu::ConstantEntry{.key = "far_end", .value = exposureFar[1]},
                                     wgpu::ConstantEntry{.key = "highlight_from", .value = exposureHighlights[0]},
                                     wgpu::ConstantEntry{.key = "highlight_to", .value = exposureHighlights[1]}};
  const wgpu::FragmentState fragment{
      .module = g_cloudModule,
      .entryPoint = exposure ? "fs_exposure" : "fs_clouds",
      .constantCount = exposure ? exposureConstants.size() : constants.size(),
      .constants = exposure ? exposureConstants.data() : constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = exposure ? "Exposure" : "Cloud shadows",
      .layout = g_cloudPipelineLayout,
      .vertex = {.module = g_cloudModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_sun_pipeline(wgpu::TextureFormat format) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_sunStrength},
      wgpu::ConstantEntry{.key = "reach", .value = g_sunReach},
      wgpu::ConstantEntry{.key = "effect_gain", .value = g_effectGain},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = nullptr, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_sunModule,
      .entryPoint = "fs_sun",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Sun shadows",
      .layout = g_cloudPipelineLayout,
      .vertex = {.module = g_sunModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

// direct: multiplied into the scene (the EFB's alpha kept), else written to a target of its own. low: the
// quarter-size pass of HD's terrain mix (the map's shadow and the grazing term, kSunLowFormat).
wgpu::RenderPipeline make_sun_map_pipeline(wgpu::TextureFormat format, bool direct, bool low = false) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_sunStrength},
      wgpu::ConstantEntry{.key = "contact", .value = g_sunMapContact},
      wgpu::ConstantEntry{.key = "debug_view", .value = low ? 0.0 : static_cast<double>(g_sunMapDebug)},
      wgpu::ConstantEntry{.key = "low_pass", .value = low ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "hd_mix", .value = hd_mix_on() && !low ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "hd_prepass", .value = g_sunHdPrepass ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "character_g", .value = g_sunSkinnedScenery ? 0.8 : 0.5},
      wgpu::ConstantEntry{.key = "sea_shadow", .value = g_sunSeaShadow >= 0.f ? g_sunSeaShadow : g_sunHdPrepass ? 0.5 : 1.0},
      wgpu::ConstantEntry{.key = "mix_e_start", .value = g_sunHdMixFades[0]},
      wgpu::ConstantEntry{.key = "mix_e_near", .value = g_sunHdMixNearPower},
      wgpu::ConstantEntry{.key = "mix_blur_start", .value = g_sunHdMixFades[1]},
      wgpu::ConstantEntry{.key = "mix_blur_end", .value = g_sunHdMixFades[2]},
      wgpu::ConstantEntry{.key = "screen_steps", .value = g_sunMapSteps ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "normal_offset", .value = g_sunMapNormalOffset},
      wgpu::ConstantEntry{.key = "slope_texels", .value = g_sunMapSlope},
      wgpu::ConstantEntry{.key = "slope_near", .value = g_sunMapSlopeNear},
      wgpu::ConstantEntry{.key = "hd_bias", .value = g_sunMapHdBias ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "normal_near", .value = g_sunMapNormalNear},
      wgpu::ConstantEntry{.key = "far_start", .value = g_sunMapFarStart},
      wgpu::ConstantEntry{.key = "far_end", .value = g_sunMapFarEnd},
      wgpu::ConstantEntry{.key = "direct", .value = direct ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "spread", .value = g_sunMapSpread},
      wgpu::ConstantEntry{.key = "smooth_terms", .value = g_sunMapSmooth ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "sea_low", .value = g_sunMapSeaLow},
      wgpu::ConstantEntry{.key = "gap_fill", .value = g_sunMapGapFill ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "taps", .value = static_cast<double>(g_sunMapTaps)},
      wgpu::ConstantEntry{.key = "sea_rise", .value = g_sunMapSeaRise},
      wgpu::ConstantEntry{.key = "sea_bias", .value = g_sunMapSeaBias},
      wgpu::ConstantEntry{.key = "soft", .value = g_sunMapSoft},
      wgpu::ConstantEntry{.key = "soft_max", .value = g_sunMapSoftMax},
      wgpu::ConstantEntry{.key = "effect_gain", .value = g_effectGain},
      wgpu::ConstantEntry{.key = "effect_debug", .value = g_effectDebug ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "facing_on", .value = g_sunFacing ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "facing_mul", .value = g_sunFacingMul},
      wgpu::ConstantEntry{.key = "facing_depth_mul", .value = g_sunFacingDepthMul},
      wgpu::ConstantEntry{.key = "facing_depth_blur", .value = g_sunFacingDepthBlur},
      wgpu::ConstantEntry{.key = "smooth_ambient", .value = g_sunSmoothAmbient ? (g_sunSmoothAmbientSkinned ? 2.0 : 1.0) : 0.0},
      wgpu::ConstantEntry{.key = "prepass_far", .value = g_sunPrepassFar ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "facing_far_start", .value = facing_far(false)},
      wgpu::ConstantEntry{.key = "facing_far_end", .value = facing_far(true)},
      wgpu::ConstantEntry{.key = "shape_on", .value = g_sunShape ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "shape_near", .value = g_sunShapeNear},
      wgpu::ConstantEntry{.key = "shape_far", .value = g_sunShapeFar},
      wgpu::ConstantEntry{.key = "ao_radius", .value = g_aoHdRadius},
      wgpu::ConstantEntry{.key = "ao_far", .value = g_aoHdFar},
      wgpu::ConstantEntry{.key = "ao_attn", .value = g_aoHdAttn},
      wgpu::ConstantEntry{.key = "ao_density", .value = g_aoHdDensity},
      wgpu::ConstantEntry{.key = "ao_offset", .value = g_aoHdOffset},
      wgpu::ConstantEntry{.key = "ao_power", .value = g_aoHdPower},
      wgpu::ConstantEntry{.key = "ao_facing", .value = g_aoHdFacing ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "ao_debug", .value = g_aoHdDebug && !low ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "cloud_scale", .value = g_cloudScale},
      wgpu::ConstantEntry{.key = "cloud_repeat", .value = g_cloudHdRepeat},
      wgpu::ConstantEntry{.key = "cloud_speed", .value = g_cloudHdSpeed},
  };
  const wgpu::BlendState multiply{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  // With HD's fog after the shade (fs_sun_map_fog): the scene times the second output, plus the first.
  const bool fog = direct && !low && g_sunFogHd && webgpu::g_dualSourceBlendingSupported;
  const wgpu::BlendState multiplyAdd{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::Src1},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{.format = low ? kSunLowFormat : format,
                                                  .blend = fog ? &multiplyAdd : direct && !low ? &multiply : nullptr,
                                                  .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_sunMapModule,
      .entryPoint = fog ? "fs_sun_map_fog" : "fs_sun_map",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Sun shadows (shadow map)",
      .layout = g_sunMapPipelineLayout,
      .vertex = {.module = g_sunMapModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_rim_pipeline(wgpu::TextureFormat format, uint32_t samples) {
  if (!g_rimModule) {
    wgpu::ShaderSourceWGSL source{};
    source.code = kRimShader;
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Rim light (HD)"};
    g_rimModule = g_device.CreateShaderModule(&moduleDescriptor);
  }
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_rimHdScale},
      wgpu::ConstantEntry{.key = "intensity", .value = g_rimHdIntensity},
      wgpu::ConstantEntry{.key = "sharpness", .value = g_rimHdSharpness},
      wgpu::ConstantEntry{.key = "rim", .value = g_rimHdRim},
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
      .blend = g_rimHdDebug == 1 ? nullptr : &add,
      .writeMask = wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue}};
  const wgpu::FragmentState fragment{
      .module = g_rimModule,
      .entryPoint = "fs_rim",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Rim light (HD)",
      .layout = g_cloudPipelineLayout,
      .vertex = {.module = g_rimModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_edge_pipeline(wgpu::TextureFormat format, uint32_t samples) {
  const std::array constants{wgpu::ConstantEntry{.key = "strength", .value = g_edgeStrength}};
  const wgpu::BlendState add{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = &add, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_edgeModule,
      .entryPoint = "fs_edge",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Edge light",
      .layout = g_cloudPipelineLayout,
      .vertex = {.module = g_edgeModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::RenderPipeline make_torch_pipeline(wgpu::TextureFormat format, uint32_t samples) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_torchStrength},
      wgpu::ConstantEntry{.key = "radius_scale", .value = g_torchRadiusScale},
  };
  const wgpu::BlendState add{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = &add, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_torchModule,
      .entryPoint = "fs_torch",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Torch light",
      .layout = g_torchPipelineLayout,
      .vertex = {.module = g_torchModule, .entryPoint = "vs_main"},
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
      wgpu::ConstantEntry{.key = "hd_ramp", .value = g_farHd ? 1.0 : 0.0},
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
      .layout = g_farPipelineLayout,
      .vertex = {.module = g_farModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

// A shadow quality's map size, cascades and filter lookups (0 low, 1 medium,
// 2 high), then each of their own variables on top. Low keeps medium's
// sharpness (at 1024 every texel term grew and small shapes became blobs) and
// saves half the lookups. Every quality
// has three cascades (out to 20,000 units): the fourth reached only islands
// deep in the haze, and its coarse texels shaded Windfall in light-blue
// blocks from across the sea, which went as it sailed into the third.
void apply_sun_quality(int level) {
  g_sunMapSize = level == 3 ? 1024 : level >= 2 ? 4096 : 2048;
  g_sunMapCascades = 3;
  g_sunMapTaps = level <= 0 ? 6 : 12;
  g_sunMapGapFill = false;
  // HD's: its map size and cascades (half its first's 1222 units across; then reaching 1395 and 14000 ahead, a
  // cascade reaching 1.5 times its range ahead of the camera) and its pre-pass.
  g_sunHdPrepass = level == 3;
  const float hdRanges[kSunMapCascades] = {611.f, 930.f, 9333.f, 0.f};
  for (uint32_t i = 0; i < kSunMapCascades; ++i)
    g_sunMapRanges[i] = level == 3 ? hdRanges[i] : 0.f;
  // DOL_AURORA_SUN_SHADOW_MAP_RANGES=a,b,c: each cascade's range in world units (local test, any quality).
  if (const char* ranges = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_RANGES")) {
    float r[3] = {};
    if (std::sscanf(ranges, "%f,%f,%f", &r[0], &r[1], &r[2]) == 3)
      for (int i = 0; i < 3; ++i)
        g_sunMapRanges[i] = std::max(r[i], 1.f);
  }
  {
    const char* follow = std::getenv("DOL_AURORA_SUN_SHADOW_CASCADES_FOLLOW");
    g_follow = follow != nullptr && follow[0] != '\0' ? follow[0] == '1' && g_sunHdPrepass : g_sunHdPrepass;
    g_followFar = env_float("DOL_AURORA_SUN_SHADOW_CASCADES_FAR", g_followFar, 1000.f, 100000.f);
    g_followSplit3 = env_float("DOL_AURORA_SUN_SHADOW_CASCADES_SPLIT3", g_followSplit3, 0.f, 100000.f);
    g_followTurnStep = env_float("DOL_AURORA_SUN_SHADOW_CASCADES_TURN_STEP", g_followTurnStep, 0.f, 90.f);
    g_followState = {};
  }
  if (const char* splits = std::getenv("DOL_AURORA_SUN_SHADOW_HD_SPLITS")) {
    float s[2] = {};
    if (std::sscanf(splits, "%f,%f", &s[0], &s[1]) == 2 && s[0] > 0.f && s[1] > s[0])
      g_sunHdSplits[0] = s[0], g_sunHdSplits[1] = s[1];
  }
  g_sunMapTaps = static_cast<uint32_t>(env_float("DOL_AURORA_SUN_SHADOW_MAP_TAPS", static_cast<float>(g_sunMapTaps), 6.f, 12.f));
  g_sunMapSize = static_cast<uint32_t>(env_float("DOL_AURORA_SUN_SHADOW_MAP_SIZE", static_cast<float>(g_sunMapSize), 256.f, 8192.f));
  g_sunMapCascades = static_cast<uint32_t>(env_float("DOL_AURORA_SUN_SHADOW_MAP_CASCADES",
                                                     static_cast<float>(g_sunMapCascades), 1.f,
                                                     static_cast<float>(kSunMapCascades)));
  if (const char* gapFill = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_GAP_FILL"); gapFill != nullptr)
    g_sunMapGapFill = gapFill[0] == '1';
}

void ensure_targets(const wgpu::Extent3D& sceneSize, wgpu::TextureFormat format, uint32_t samples) {
  // A quality set while running: its map is made at its size when next used
  // (sun_map_view), and the two pipelines that take its lookups and gap fill
  // as constants are made again below.
  if (const int level = g_sunQualityPending.exchange(-1, std::memory_order_relaxed); level >= 0) {
    apply_sun_quality(level);
    g_sunMapPass = {};
    g_sunMapDirect = {};
    g_sunMapLow = {};
  }
  if (format != g_format) {
    g_format = format;
    g_bright = make_pipeline("Bloom bright pass", "fs_bright", format, 1, kReplace);
    g_blurH = make_pipeline("Bloom blur (horizontal)", "fs_blur_h", format, 1, kReplace);
    g_blurV = make_pipeline("Bloom blur (vertical)", "fs_blur_v", format, 1, kReplace);
    g_down = make_pipeline("Far blur downsample", "fs_down", format, 1, kReplace);
    g_aoPass = make_ao_pipeline("Ambient occlusion", "fs_ao", format, 1, kReplace);
    g_aoBlurH = make_ao_pipeline("Ambient occlusion blur (horizontal)", "fs_ao_blur_h", format, 1, kReplace);
    g_aoBlurV = make_ao_pipeline("Ambient occlusion blur (vertical)", "fs_ao_blur_v", format, 1, kReplace);
    g_cloudPass = {};
    g_sunPass = {};
    g_sunMapPass = {};
    g_sunMapDirect = {};
    g_sunMapLow = {};
    g_edgePass = {};
    g_torchPass = {};
    g_composite = {};
    g_farComposite = {};
    g_exposureDepth = {};
    g_aoApply = {};
    g_size = {};
  }
  if (!g_composite || samples != g_compositeSamples) {
    g_composite = make_pipeline("Bloom composite", "fs_composite", format, samples, kScreen);
    g_compositeSamples = samples;
  }
  if (!g_sunPass) {
    g_sunPass = make_sun_pipeline(format);
  }
  if (!g_sunMapDirect)
    g_sunMapDirect = make_sun_map_pipeline(format, true);
  if (!g_sunMapPass) {
    g_sunMapPass = make_sun_map_pipeline(format, false);
  }
  if (!g_torchPass || samples != g_torchSamples) {
    g_torchPass = make_torch_pipeline(format, samples);
    g_torchSamples = samples;
  }
  if (!g_edgePass || samples != g_edgeSamples) {
    g_edgePass = make_edge_pipeline(format, samples);
    g_edgeSamples = samples;
  }
  if (!g_cloudPass || samples != g_cloudSamples) {
    g_cloudPass = make_cloud_pipeline(format, samples);
    g_cloudSamples = samples;
  }
  if (!g_aoApply || samples != g_aoSamples) {
    g_aoApply = make_pipeline("Sun shadows apply", "fs_ao_apply", format, samples,
                              g_aoDebug == 1 ? kReplace : kMultiply);
    g_aoUpscale = make_ao_pipeline("Ambient occlusion apply", "fs_ao_upscale", format, samples,
                                   g_aoDebug == 1 ? kReplace : kMultiply);
    g_aoSamples = samples;
  }
  if (!g_farComposite || samples != g_farSamples) {
    g_farComposite = make_far_pipeline(format, samples);
    g_exposureDepth = make_cloud_pipeline(format, samples, true);
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
    g_aoTex[i] = g_device.CreateTexture(&descriptor);
    g_aoView[i] = g_aoTex[i].CreateView();
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
// effects: the effects' mask (effect_mask_target()) for the occlusion's apply, else a black texel.
void draw_depth(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                const wgpu::TextureView& source, const wgpu::TextureView& depthView, const wgpu::TextureView& target,
                const wgpu::TextureView& resolve, bool load, const wgpu::TextureView& effects = nullptr) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = source},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .textureView = effects ? effects : g_effectNoneView},
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
      .loadOp = load ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {1.0, 1.0, 1.0, 1.0},
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
void draw_with_uniform(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                       const wgpu::BindGroupLayout& layout, const void* uniform, size_t uniformSize,
                       const wgpu::TextureView& depthView, const wgpu::TextureView& target,
                       const wgpu::TextureView& resolve, const wgpu::TextureView* texture = nullptr,
                       const wgpu::TextureView* scene = nullptr);

void draw_with_view(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                    const std::array<float, 20>& view, const wgpu::TextureView& depthView,
                    const wgpu::TextureView& target, const wgpu::TextureView& resolve,
                    const wgpu::TextureView* texture = nullptr, const wgpu::TextureView* scene = nullptr) {
  draw_with_uniform(cmd, label, pipeline, g_cloudLayout, view.data(), sizeof(view), depthView, target, resolve,
                    texture, scene);
}

void draw_with_uniform(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                       const wgpu::BindGroupLayout& layout, const void* uniform, size_t uniformSize,
                       const wgpu::TextureView& depthView, const wgpu::TextureView& target,
                       const wgpu::TextureView& resolve, const wgpu::TextureView* texture,
                       const wgpu::TextureView* scene) {
  // A small buffer for each frame, so a frame still in flight keeps its camera.
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Post effects view",
      .usage = wgpu::BufferUsage::Uniform,
      .size = uniformSize,
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, uniformSize), uniform, uniformSize);
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = texture != nullptr ? *texture : g_farView[0]},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .buffer = buffer, .offset = 0, .size = uniformSize},
      wgpu::BindGroupEntry{.binding = 4, .textureView = scene != nullptr ? *scene : g_farView[0]},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = layout,
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
// The sun shadows from the shadow map (kSunMapShader) into `target`.
// direct: multiplied into `target`, the scene (with `resolve`), else drawn into a target of its own.
// HD's screen-space blur of its shadow buffer (agl image_filter.sh, kernel 9 "omit"): three bilinear taps at 0
// and +-1.3636 texels, weighted 5/17 and 6/17, one axis a pass.
constexpr const char* kSunBlurShader = R"""(
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
override axis_x: f32 = 1.0;
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
fn fs_blur(in: VertexOutput) -> @location(0) vec4<f32> {
    let o = vec2<f32>(axis_x, 1.0 - axis_x) * 1.3636 / vec2<f32>(textureDimensions(src));
    return 0.29411765 * textureSampleLevel(src, samp, in.uv, 0.0) +
           0.35294118 * (textureSampleLevel(src, samp, in.uv - o, 0.0) + textureSampleLevel(src, samp, in.uv + o, 0.0));
}
// One bilinear tap: a smaller or larger copy (the characters' shadow pre-pass's mip 1, as HD makes it).
@fragment
fn fs_copy(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, samp, in.uv, 0.0);
}
)""";

void make_sun_blur_pipelines() {
  wgpu::ShaderSourceWGSL source{};
  source.code = kSunBlurShader;
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Sun shadows' blur"};
  g_sunBlurModule = g_device.CreateShaderModule(&moduleDescriptor);
  const std::array entries{
      wgpu::BindGroupLayoutEntry{.binding = 0, .visibility = wgpu::ShaderStage::Fragment,
                                 .sampler = {.type = wgpu::SamplerBindingType::Filtering}},
      wgpu::BindGroupLayoutEntry{.binding = 1, .visibility = wgpu::ShaderStage::Fragment,
                                 .texture = {.sampleType = wgpu::TextureSampleType::Float,
                                             .viewDimension = wgpu::TextureViewDimension::e2D}},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "Sun shadows' blur", .entryCount = entries.size(), .entries = entries.data()};
  g_sunBlurLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                .bindGroupLayouts = &g_sunBlurLayout};
  g_sunBlurPipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  for (int axis = 0; axis < 2; ++axis) {
    const wgpu::ConstantEntry constant{.key = "axis_x", .value = axis == 0 ? 1.0 : 0.0};
    const wgpu::ColorTargetState target{.format = kSunLowFormat, .writeMask = wgpu::ColorWriteMask::All};
    const wgpu::FragmentState fragment{.module = g_sunBlurModule,
                                       .entryPoint = "fs_blur",
                                       .constantCount = 1,
                                       .constants = &constant,
                                       .targetCount = 1,
                                       .targets = &target};
    const wgpu::RenderPipelineDescriptor descriptor{
        .label = axis == 0 ? "Sun shadows' blur (X)" : "Sun shadows' blur (Y)",
        .layout = g_sunBlurPipelineLayout,
        .vertex = {.module = g_sunBlurModule, .entryPoint = "vs_main"},
        .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
        .multisample = {.count = 1, .mask = UINT32_MAX},
        .fragment = &fragment,
    };
    (axis == 0 ? g_sunBlurX : g_sunBlurY) = g_device.CreateRenderPipeline(&descriptor);
  }
  const wgpu::ColorTargetState target{.format = kSunLowFormat, .writeMask = wgpu::ColorWriteMask::All};
  const wgpu::FragmentState fragment{
      .module = g_sunBlurModule, .entryPoint = "fs_copy", .targetCount = 1, .targets = &target};
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Sun shadows' copy",
      .layout = g_sunBlurPipelineLayout,
      .vertex = {.module = g_sunBlurModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  g_sunCopy = g_device.CreateRenderPipeline(&descriptor);
}

// The pre-pass blended with the last frame's (g_sunTemporal): each place in the world takes back what it had last
// frame (found through last frame's camera), kept within what this frame's neighbours hold so a moving shadow
// doesn't leave a trail, by `range.z` from `range.y` units away and none nearer than `range.x`. Red (the shadow)
// and blue (HD's SSAO) are blended; green (the grazing term) is this frame's. Alpha keeps the depth for the next.
constexpr const char* kSunTemporalShader = R"""(
// DOL_AURORA_SUN_SHADOW_TEMPORAL_DEBUG=1 (local test): no blend; green is 1 - |this frame's shadow - the last frame's
// at the same place in the world| (1 where there was no last frame there), for the main pass's debug view 12: a
// shadow that changed with the camera's motion taken out (cascades redrawn on a new grid, a split moving).
override debug_diff: f32 = 0.0;
struct Temporal {
    rows: array<vec4<f32>, 3>,  // this frame's view to world, by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, near / (far - near)
    prev: array<vec4<f32>, 3>,  // last frame's world to view, by rows
    prev_lens: vec4<f32>,       // its tan(fovy / 2), aspect; z: 1 when there is a last frame; w: game frames since
    range: vec4<f32>,           // nearest blended, fully blended from, the most of the last frame's
};
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var current: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d<f32>;
@group(0) @binding(3) var depth: texture_depth_2d;
@group(0) @binding(4) var<uniform> t: Temporal;
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
fn fs_temporal(in: VertexOutput) -> @location(0) vec4<f32> {
    let cur = textureSampleLevel(current, samp, in.uv, 0.0);
    let dsize = vec2<f32>(textureDimensions(depth));
    let d = textureLoad(depth, clamp(vec2<i32>(in.uv * dsize), vec2<i32>(0), vec2<i32>(dsize) - vec2<i32>(1)), 0);
    let z = select(t.lens.z / (d + t.lens.w), 0.0, d <= 0.0);
    if (z <= 0.0 || t.prev_lens.z < 0.5) {
        return vec4(cur.r, select(cur.g, 1.0, debug_diff > 0.5), cur.b, z);
    }
    let ndc = vec2(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let p = vec4(vec3(ndc.x * t.lens.x * t.lens.y, ndc.y * t.lens.x, -1.0) * z, 1.0);
    let w = vec4(dot(t.rows[0], p), dot(t.rows[1], p), dot(t.rows[2], p), 1.0);
    let q = vec3(dot(t.prev[0], w), dot(t.prev[1], w), dot(t.prev[2], w));
    let pz = -q.z;
    if (pz <= 0.0) {
        return vec4(cur.r, select(cur.g, 1.0, debug_diff > 0.5), cur.b, z);
    }
    let pn = vec2(q.x / (pz * t.prev_lens.x * t.prev_lens.y), q.y / (pz * t.prev_lens.x));
    let puv = vec2(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
    if (any(puv < vec2(0.0)) || any(puv > vec2(1.0))) {
        return vec4(cur.r, select(cur.g, 1.0, debug_diff > 0.5), cur.b, z);
    }
    let hist = textureSampleLevel(history, samp, puv, 0.0);
    // Hidden last frame: something else was there (its depth differs).
    if (abs(hist.a - pz) > 0.03 * pz + 2.0) {
        return vec4(cur.r, select(cur.g, 1.0, debug_diff > 0.5), cur.b, z);
    }
    if (debug_diff > 0.5) {
        return vec4(cur.r, 1.0 - abs(cur.r - hist.r), cur.b, z);
    }
    let size = vec2<i32>(textureDimensions(current));
    let c = vec2<i32>(in.uv * vec2<f32>(size));
    var lo = cur.rgb;
    var hi = cur.rgb;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let s = textureLoad(current, clamp(c + vec2(dx, dy), vec2<i32>(0), size - vec2<i32>(1)), 0).rgb;
            lo = min(lo, s);
            hi = max(hi, s);
        }
    }
    // range.z is what the last frame keeps a thirtieth of a second on (one game frame); prev_lens.w holds how many
    // of those since, so Smooth Motion's in-between frames blend as much over a game frame as one frame without it.
    let a = pow(t.range.z, t.prev_lens.w) * smoothstep(t.range.x, t.range.y, z);
    let blended = mix(cur.rgb, clamp(hist.rgb, lo, hi), a);
    return vec4(blended.r, cur.g, blended.b, z);
}
)""";

void make_sun_temporal_pipeline() {
  wgpu::ShaderSourceWGSL source{};
  source.code = kSunTemporalShader;
  const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Sun shadows' blend"};
  g_sunTemporalModule = g_device.CreateShaderModule(&moduleDescriptor);
  const wgpu::TextureBindingLayout colour{.sampleType = wgpu::TextureSampleType::Float,
                                          .viewDimension = wgpu::TextureViewDimension::e2D};
  const std::array entries{
      wgpu::BindGroupLayoutEntry{.binding = 0, .visibility = wgpu::ShaderStage::Fragment,
                                 .sampler = {.type = wgpu::SamplerBindingType::Filtering}},
      wgpu::BindGroupLayoutEntry{.binding = 1, .visibility = wgpu::ShaderStage::Fragment, .texture = colour},
      wgpu::BindGroupLayoutEntry{.binding = 2, .visibility = wgpu::ShaderStage::Fragment, .texture = colour},
      wgpu::BindGroupLayoutEntry{.binding = 3, .visibility = wgpu::ShaderStage::Fragment,
                                 .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                                             .viewDimension = wgpu::TextureViewDimension::e2D}},
      wgpu::BindGroupLayoutEntry{.binding = 4, .visibility = wgpu::ShaderStage::Fragment,
                                 .buffer = {.type = wgpu::BufferBindingType::Uniform}},
  };
  const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
      .label = "Sun shadows' blend", .entryCount = entries.size(), .entries = entries.data()};
  g_sunTemporalLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
  const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                .bindGroupLayouts = &g_sunTemporalLayout};
  g_sunTemporalPipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  const wgpu::ColorTargetState target{.format = kSunLowFormat, .writeMask = wgpu::ColorWriteMask::All};
  static const bool debugDiff = [] {
    const char* env = std::getenv("DOL_AURORA_SUN_SHADOW_TEMPORAL_DEBUG");
    return env != nullptr && env[0] == '1';
  }();
  const wgpu::ConstantEntry constant{.key = "debug_diff", .value = debugDiff ? 1.0 : 0.0};
  const wgpu::FragmentState fragment{.module = g_sunTemporalModule,
                                     .entryPoint = "fs_temporal",
                                     .constantCount = 1,
                                     .constants = &constant,
                                     .targetCount = 1,
                                     .targets = &target};
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Sun shadows' blend",
      .layout = g_sunTemporalPipelineLayout,
      .vertex = {.module = g_sunTemporalModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  g_sunTemporalPipeline = g_device.CreateRenderPipeline(&descriptor);
}

void draw_into(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
               const wgpu::BindGroup& group, const wgpu::TextureView& target);

// The blend (g_sunTemporal): this frame's pre-pass (`raw`, in g_sunLow[1]'s mip 0) with the last frame's into
// g_sunLow[0]'s mip 0, then that copied as the next frame's last. `nearOverFar`: near / (far - near).
void draw_sun_temporal(const wgpu::CommandEncoder& cmd, const Camera& camera, const wgpu::TextureView& depthView,
                       float nearOverFar) {
  if (!g_sunTemporalPipeline)
    make_sun_temporal_pipeline();
  std::array<float, 36> u{};
  std::memcpy(u.data(), camera.view.data(), 12 * sizeof(float));
  u[12] = camera.view[12];
  u[13] = camera.view[13];
  u[14] = camera.view[14];
  u[15] = nearOverFar;
  // Last frame's world to view: the inverse of its view to world (rows of [A | b]).
  const float* m = g_sunHistCamera.data();
  const float a00 = m[0], a01 = m[1], a02 = m[2], a10 = m[4], a11 = m[5], a12 = m[6], a20 = m[8], a21 = m[9],
              a22 = m[10];
  const float det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
  const bool valid = g_sunHistValid && std::fabs(det) > 1e-12f;
  if (valid) {
    const float inv[9] = {(a11 * a22 - a12 * a21) / det, (a02 * a21 - a01 * a22) / det, (a01 * a12 - a02 * a11) / det,
                          (a12 * a20 - a10 * a22) / det, (a00 * a22 - a02 * a20) / det, (a02 * a10 - a00 * a12) / det,
                          (a10 * a21 - a11 * a20) / det, (a01 * a20 - a00 * a21) / det, (a00 * a11 - a01 * a10) / det};
    const float b[3] = {m[3], m[7], m[11]};
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c)
        u[16 + r * 4 + c] = inv[r * 3 + c];
      u[16 + r * 4 + 3] = -(inv[r * 3] * b[0] + inv[r * 3 + 1] * b[1] + inv[r * 3 + 2] * b[2]);
    }
  }
  u[28] = m[12];
  u[29] = m[13];
  u[30] = valid ? 1.f : 0.f;
  // Game frames (thirtieths of a second) since the last blend, from the camera's seconds (each of Smooth Motion's
  // in-between frames has its own); at least a quarter, so a frame whose time didn't move still takes some of its own.
  const float since = camera.view[16] - g_sunHistTime;
  u[31] = std::clamp(since * 30.f, 0.25f, 4.f);
  u[32] = g_sunTemporalRange[0];
  u[33] = g_sunTemporalRange[1];
  u[34] = g_sunTemporalRange[2];
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Sun shadows' blend", .usage = wgpu::BufferUsage::Uniform, .size = sizeof(u), .mappedAtCreation = true};
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, sizeof(u)), u.data(), sizeof(u));
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = g_sunLowMipView[1][0]},
      wgpu::BindGroupEntry{.binding = 2, .textureView = g_sunHistView},
      wgpu::BindGroupEntry{.binding = 3, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 4, .buffer = buffer, .offset = 0, .size = sizeof(u)},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .layout = g_sunTemporalLayout, .entryCount = entries.size(), .entries = entries.data()};
  draw_into(cmd, "Sun shadows' blend", g_sunTemporalPipeline, g_device.CreateBindGroup(&groupDescriptor),
            g_sunLowMipView[0][0]);
  const wgpu::TexelCopyTextureInfo from{.texture = g_sunLow[0], .mipLevel = 0};
  const wgpu::TexelCopyTextureInfo to{.texture = g_sunHist};
  const wgpu::Extent3D extent{g_sunLowSize.width, g_sunLowSize.height, 1};
  cmd.CopyTextureToTexture(&from, &to, &extent);
  std::memcpy(g_sunHistCamera.data(), camera.view.data(), 16 * sizeof(float));
  g_sunHistTime = camera.view[16];
  g_sunHistValid = true;
}

// HD's SSAO runs this frame (in the terrain mix), in place of BlueWake's own.
bool ao_hd_on() noexcept { return g_aoHd && hd_mix_on() && ambient_occlusion_enabled() && sun_shadow_map_enabled(); }

// The cloud shadows on HD's rule run this frame, in the sun shadows' pass in place of their own.
bool clouds_hd_on() noexcept { return g_cloudHd && cloud_shadows_enabled() && sun_shadow_map_enabled(); }

// The cloud shadows' floor on HD's rule (cCloudShadowPower; cloud = clamp(texel + floor)): DOL_AURORA_CLOUD_SHADOWS_HD_
// POWER, else the host's from the game's clock (aurora_set_cloud_hd_floor: hd/d/d_kankyo_3.cpp setSunpos's
// 1 - max(0.25, -0.9 cos(t - 15 deg)), 0.364 at 10:00, 0.1307 at noon, and 0.75 from a low sun through the night,
// when HD's clouds still take a quarter of the light), else 1 - max(0.25, 0.9 x the light's height). That last is
// HD's only with the true sun's height: with HD's light rule (aurora_set_sun_hd_light) the light is held at 40,000
// high before 09:00 and after 17:00, and gave 0.596 at 07:30 and 18:30 where HD has 0.75. The scenery (uniform[87])
// and the characters (actor_cloud_source()) both take it from here.
float cloud_hd_floor(const Camera& camera) noexcept {
  if (g_cloudHdPower >= 0.f)
    return g_cloudHdPower;
  const float host = g_cloudHdFloor.load(std::memory_order_relaxed);
  if (host >= 0.f)
    return host;
  return 1.f - std::max(0.25f, 0.9f * std::clamp(camera.view[18], 0.f, 1.f));
}

// A black texel in the effects' mask's place (no effects), made on first use.
const wgpu::TextureView& effect_none_view() {
  if (!g_effectNone) {
    const wgpu::TextureDescriptor descriptor{
        .label = "No effects",
        .usage = wgpu::TextureUsage::TextureBinding,
        .size = {1, 1, 1},
        .format = wgpu::TextureFormat::RGBA8Unorm,
    };
    g_effectNone = g_device.CreateTexture(&descriptor);
    g_effectNoneView = g_effectNone.CreateView();
  }
  return g_effectNoneView;
}

// The quarter-size targets of HD's terrain mix, for a scene of `size` (half the size in HD's pass, as its pre-pass).
void ensure_sun_low(const wgpu::Extent3D& size) {
  const uint32_t div = g_sunHdPrepass ? 2u : 4u;
  const wgpu::Extent3D low{std::max(1u, size.width / div), std::max(1u, size.height / div), 1};
  if (g_sunLow[0] && g_sunLowSize.width == low.width && g_sunLowSize.height == low.height)
    return;
  for (int i = 0; i < 2; ++i) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Sun shadows (quarter size)",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding |
                 wgpu::TextureUsage::CopySrc,
        .size = low,
        .format = kSunLowFormat,
        .mipLevelCount = 2,
    };
    g_sunLow[i] = g_device.CreateTexture(&descriptor);
    g_sunLowView[i] = g_sunLow[i].CreateView();
    for (uint32_t mip = 0; mip < 2; ++mip) {
      const wgpu::TextureViewDescriptor one{.baseMipLevel = mip, .mipLevelCount = 1};
      g_sunLowMipView[i][mip] = g_sunLow[i].CreateView(&one);
    }
  }
  // The blend's last frame (g_sunTemporal): the pre-pass after the blend, copied each frame, its depth in alpha.
  const wgpu::TextureDescriptor history{
      .label = "Sun shadows (last frame)",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = low,
      .format = kSunLowFormat,
  };
  g_sunHist = g_device.CreateTexture(&history);
  g_sunHistView = g_sunHist.CreateView();
  g_sunHistValid = false;
  g_sunLowSize = low;
}

// One full-screen draw into `target` (cleared first).
void draw_into(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
               const wgpu::BindGroup& group, const wgpu::TextureView& target) {
  const wgpu::RenderPassColorAttachment attachment{
      .view = target, .loadOp = wgpu::LoadOp::Clear, .storeOp = wgpu::StoreOp::Store, .clearValue = {1.0, 0.0, 0.0, 1.0}};
  const wgpu::RenderPassDescriptor descriptor{.label = label, .colorAttachmentCount = 1, .colorAttachments = &attachment};
  const auto pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, group, 0, nullptr);
  pass.Draw(3);
  pass.End();
}

// The sun shadow map pass's uniform and what its bind group takes (draw_sun_map(), and the characters' own
// shadow pre-pass, draw_actor_shadow(): `actor` leaves HD's SSAO out of its low pass, which they don't read).
struct SunMapInputs {
  wgpu::Buffer buffer;
  bool mask = false;
  bool normals = false;
  float farTerm = 0.f; // near / (far - near)
};

SunMapInputs sun_map_inputs(const Camera& camera, const SunMap& sunMap, bool actor) {
  std::array<float, kSunMapUniformFloats> uniform{};
  std::memcpy(uniform.data(), camera.view.data(), sizeof(camera.view));
  for (uint32_t cascade = 0; cascade < kSunMapCascades; ++cascade)
    std::memcpy(uniform.data() + 20 + cascade * 12, sunMap.world[cascade].data(), sizeof(sunMap.world[cascade]));
  uniform[68] = 1.f / static_cast<float>(g_sunMapSize);
  uniform[69] = static_cast<float>(sunMap.cascades);
  uniform[70] = sunMap.fade;
  // The far plane's share of the depth (view_depth), 0 for one infinitely far.
  const float nearPlane = camera.view[14];
  uniform[71] = camera.farPlane > nearPlane ? nearPlane / (camera.farPlane - nearPlane) : 0.f;
  // The map's light (the moon at night) in the place of the sun.
  std::memcpy(uniform.data() + 17, sunMap.light.data(), sizeof(sunMap.light));
  for (uint32_t cascade = 0; cascade < kSunMapCascades; ++cascade) {
    // The bias (world units for a cascade 5000 across) follows each
    // cascade's texels: 3 near Link's feet, 12, 48 and 192 further out, in
    // each one's own depth's units.
    const float range = cascade_range(cascade);
    uniform[72 + cascade] = g_sunMapBias * (range / 5000.f) / (2.f * cascade_depth_of(cascade));
  }
  // The cascades that follow the camera: on, where the fourth starts (0: none), the far fade.
  uniform[96] = g_followState.on ? 1.f : 0.f;
  uniform[97] = g_followState.on ? g_followState.split3 : 0.f;
  uniform[98] = g_followState.fadeStart;
  uniform[99] = g_followState.fadeEnd;
  // The sea mask is drawn when it is wanted (common.cpp encode_sea_mask, sea_mask_wanted()): the facing
  // term needs it to leave characters and the sea out, and is skipped without it.
  const bool mask = g_seaMaskView && sea_mask_wanted();
  // DOL_AURORA_SUN_SHADOW_FACING_LOG=1 (local debug): whether the facing term can run, a line every 60 frames.
  static const bool facingLog = std::getenv("DOL_AURORA_SUN_SHADOW_FACING_LOG") != nullptr;
  static uint32_t facingLogCount = 0;
  if (facingLog && facingLogCount++ % 60 == 0)
    std::fprintf(stderr, "[sun-facing] on=%d mask view=%d exposure=%d\n", g_sunFacing ? 1 : 0, g_seaMaskView ? 1 : 0,
                 exposure_active() ? 1 : 0);
  uniform[76] = mask ? 1.f : 0.f;
  // HD's SSAO (made in the terrain mix's quarter-size pass) and the cloud shadows on HD's rule.
  uniform[79] = ao_hd_on() ? 1.f : 0.f;
  uniform[83] = clouds_hd_on() ? 1.f : 0.f;
  // HD's cloud texture, loaded here when the host has given its path since (aurora_set_hd_cloud_texture).
  {
    std::string path;
    {
      std::lock_guard lock{g_cameraMutex};
      if (g_cloudTexPending) {
        path = g_cloudTexPath;
        g_cloudTexPending = false;
      } else {
        path = "\x01";
      }
    }
    if (path != "\x01") {
      g_cloudTexView = {};
      g_cloudTex = {};
      if (!path.empty())
        load_cloud_texture(path.c_str());
    }
  }
  uniform[91] = g_cloudTexView ? 1.f : 0.f;
  // HD's splits this frame, as its first two cascades are sized (cascade_range).
  uniform[92] = g_sunHdPrepass ? hd_split(0) : g_sunHdSplits[0];
  uniform[93] = g_sunHdPrepass ? hd_split(1) : g_sunHdSplits[1];
  uniform[87] = cloud_hd_floor(camera);
  // The scenery's smooth normals (sun_normal_target()), drawn with the mask while the facing term is on.
  const bool normals = mask && g_sunNormalView && sun_normals_wanted();
  uniform[78] = normals ? 1.f : 0.f;
  // Wind Waker HD's character light took the toon characters' shade in their draws: their pixels are left as drawn.
  uniform[94] = actor_light_applied() ? 1.f : 0.f;
  // Wind Waker HD's light buffer on the lit scenery, for the shade to leave (fs_sun_map's point_sum).
  if (point_lights_hd_active() && actor_light_applied() && camera.lightCount > 0) {
    uniform[100] = static_cast<float>(camera.lightCount);
    uniform[101] = 1.25f;
    uniform[102] = 1.f;
    std::memcpy(uniform.data() + 104, camera.lights.data(), camera.lightCount * 8 * sizeof(float));
  }
  // The lit scenery's fog, for the shade to fog after itself (fs_sun_map_fog).
  if (g_sunFogHd && !actor) {
    std::lock_guard lock{g_sceneFogMutex};
    if (g_sceneFog.on && g_sceneFog.stale++ < 8) {
      constexpr size_t f = 104 + kMaxLights * 8;
      uniform[f + 0] = g_sceneFog.a;
      uniform[f + 1] = g_sceneFog.b;
      uniform[f + 2] = g_sceneFog.shift;
      uniform[f + 3] = g_sceneFog.c;
      for (int i = 0; i < 3; ++i)
        uniform[f + 4 + i] = g_sceneFog.colour[i];
      uniform[f + 7] = g_sceneFog.square ? 2.f : 1.f;
      uniform[f + 8] = g_sceneFog.range ? 1.f : 0.f;
      uniform[f + 9] = g_sceneFog.centre;
      uniform[f + 10] = g_sceneFog.width;
      for (int i = 0; i < 12; ++i)
        uniform[f + 12 + i] = g_sceneFog.table[i];
      for (int i = 0; i < 16; ++i)
        uniform[f + 24 + i] = g_fogHdStarts[i];
    }
  }
  // The full shadow's colour, or Wind Waker HD's light for the hour when the host gives it.
  {
    std::lock_guard lock{g_cameraMutex};
    uniform[77] = g_sunPalette.on ? 1.f : 0.f;
    for (int i = 0; i < 3; ++i) {
      uniform[80 + i] = g_sunPalette.on ? g_sunPalette.sky[i] : g_sunShadowColour[i];
      uniform[84 + i] = g_sunPalette.ground[i];
      uniform[88 + i] = g_sunPalette.lit[i];
    }
  }
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Sun shadow map view",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(uniform),
      .mappedAtCreation = true,
  };
  if (actor)
    uniform[79] = 0.f;
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, sizeof(uniform)), uniform.data(), sizeof(uniform));
  buffer.Unmap();
  return {.buffer = buffer, .mask = mask, .normals = normals, .farTerm = uniform[71]};
}

// effects: the effects' mask (effect_mask_target()), whose effects the shade leaves.
wgpu::BindGroup sun_map_group(const SunMapInputs& in, const wgpu::TextureView& depthView,
                              const wgpu::TextureView& blurred, const wgpu::TextureView& effects) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = in.mask ? g_seaMaskView : g_farView[0]},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .buffer = in.buffer, .offset = 0,
                           .size = kSunMapUniformFloats * sizeof(float)},
      wgpu::BindGroupEntry{.binding = 4, .textureView = g_sunMapView},
      wgpu::BindGroupEntry{.binding = 5, .sampler = g_sunMapCompare},
      wgpu::BindGroupEntry{.binding = 6, .textureView = in.normals ? g_sunNormalView : g_farView[0]},
      wgpu::BindGroupEntry{.binding = 7, .textureView = blurred},
      wgpu::BindGroupEntry{.binding = 8, .textureView = g_cloudTexView ? g_cloudTexView : g_farView[0]},
      wgpu::BindGroupEntry{.binding = 9, .sampler = g_cloudSampler},
      wgpu::BindGroupEntry{.binding = 10, .textureView = effects},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_sunMapLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  return g_device.CreateBindGroup(&bindGroupDescriptor);
}

// effects: the effects' mask (effect_mask_target()), whose effects it leaves.
void draw_sun_map(const wgpu::CommandEncoder& cmd, const Camera& camera, const SunMap& sunMap,
                  const wgpu::TextureView& depthView, const wgpu::TextureView& effects,
                  const wgpu::TextureView& target, bool direct, const wgpu::Extent3D& size,
                  const wgpu::TextureView& resolve = nullptr) {
  const SunMapInputs inputs = sun_map_inputs(camera, sunMap, false);
  const auto sun_group = [&](const wgpu::TextureView& blurred) {
    return sun_map_group(inputs, depthView, blurred, effects);
  };
  // HD's terrain mix: the map's shadow and the grazing term at a quarter of the size, blurred (X then Y,
  // g_sunHdMixBlurs times), for the main pass to take by distance and grazing.
  wgpu::TextureView blurred = g_farView[0];
  if (hd_mix_on()) {
    ensure_sun_low(size);
    if (!g_sunBlurX)
      make_sun_blur_pipelines();
    if (!g_sunMapLow)
      g_sunMapLow = make_sun_map_pipeline(kSunLowFormat, false, true);
    // With the blend, the pre-pass goes to g_sunLow[1] first and the blend writes g_sunLow[0]. A monotone frame's
    // second lighting (encode_late()) takes none: the frame's blend is done, and its history stays the frame's.
    const bool temporal = g_sunTemporal && !g_sunTemporalHeld;
    draw_into(cmd, "Sun shadows (quarter size)", g_sunMapLow, sun_group(g_farView[0]),
              g_sunLowMipView[temporal ? 1 : 0][0]);
    if (temporal)
      draw_sun_temporal(cmd, camera, depthView, inputs.farTerm);
    else if (!g_sunTemporalHeld)
      g_sunHistValid = false;
    const auto blur_group = [&](const wgpu::TextureView& view) {
      const std::array entries{wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
                               wgpu::BindGroupEntry{.binding = 1, .textureView = view}};
      const wgpu::BindGroupDescriptor descriptor{
          .layout = g_sunBlurLayout, .entryCount = entries.size(), .entries = entries.data()};
      return g_device.CreateBindGroup(&descriptor);
    };
    for (int i = 0; i < g_sunHdMixBlurs; ++i) {
      draw_into(cmd, "Sun shadows' blur (X)", g_sunBlurX, blur_group(g_sunLowMipView[0][0]), g_sunLowMipView[1][0]);
      draw_into(cmd, "Sun shadows' blur (Y)", g_sunBlurY, blur_group(g_sunLowMipView[1][0]), g_sunLowMipView[0][0]);
    }
    // HD's pass: its pre-pass's mip 1, half the size again and blurred once more (HD's farther and grazing read).
    if (g_sunHdPrepass) {
      draw_into(cmd, "Sun shadows' mip 1 (X)", g_sunBlurX, blur_group(g_sunLowMipView[0][0]), g_sunLowMipView[1][1]);
      draw_into(cmd, "Sun shadows' mip 1 (Y)", g_sunBlurY, blur_group(g_sunLowMipView[1][1]), g_sunLowMipView[0][1]);
    }
    blurred = g_sunLowView[0];
    g_sunLowEncode = g_postEncodes;
  }
  const auto bindGroup = sun_group(blurred);
  const std::array attachments{wgpu::RenderPassColorAttachment{
      .view = target,
      .resolveTarget = direct ? resolve : nullptr,
      .loadOp = direct ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {1.0, 1.0, 1.0, 1.0},
  }};
  const wgpu::RenderPassDescriptor descriptor{
      .label = "Sun shadows (shadow map)",
      .colorAttachmentCount = attachments.size(),
      .colorAttachments = attachments.data(),
      .timestampWrites = webgpu::gpu_prof::pass_writes("Sun shadows (shadow map)"),
  };
  const auto pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(direct ? g_sunMapDirect : g_sunMapPass);
  pass.SetBindGroup(0, bindGroup, 0, nullptr);
  pass.Draw(3);
  pass.End();
}

void ensure_actor_low(const wgpu::Extent3D& size) {
  const wgpu::Extent3D half{std::max(1u, size.width / 2), std::max(1u, size.height / 2), 1};
  if (g_actorLow && g_actorLowSize.width == half.width && g_actorLowSize.height == half.height)
    return;
  const wgpu::TextureUsage usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
  const wgpu::TextureDescriptor descriptor{
      .label = "Characters' shadow (half size)", .usage = usage, .size = half, .format = kSunLowFormat, .mipLevelCount = 2};
  g_actorLow = g_device.CreateTexture(&descriptor);
  g_actorLowView = g_actorLow.CreateView();
  for (uint32_t mip = 0; mip < 2; ++mip) {
    const wgpu::TextureViewDescriptor one{.baseMipLevel = mip, .mipLevelCount = 1};
    g_actorLowMip[mip] = g_actorLow.CreateView(&one);
  }
  const wgpu::Extent3D eighth{std::max(1u, size.width / 8), std::max(1u, size.height / 8), 1};
  for (int i = 0; i < 3; ++i) {
    const wgpu::TextureDescriptor tmp{
        .label = "Characters' shadow (blur)", .usage = usage, .size = i == 0 ? half : eighth, .format = kSunLowFormat};
    g_actorTmp[i] = g_device.CreateTexture(&tmp);
    g_actorTmpView[i] = g_actorTmp[i].CreateView();
  }
  g_actorLowSize = half;
}

bool draw_actor_shadow_now(const wgpu::CommandEncoder& cmd, const Camera& camera, const SunMap& sunMap,
                           const wgpu::Extent3D& size, wgpu::TextureFormat format) {
  g_actorLowReady = false;
  if (!g_module || !camera.valid || !sunMap.valid || !sun_shadows_enabled() || !g_seaMaskDepthView ||
      g_seaMaskSize.width != size.width || g_seaMaskSize.height != size.height || size.width < 16 ||
      size.height < 16)
    return false;
  ensure_targets(size, format, 1);
  if (!g_sunMapView)
    return false;
  const webgpu::gpu_prof::Zone zone{cmd, "Characters' shadow"};
  ensure_actor_low(size);
  if (!g_sunBlurX)
    make_sun_blur_pipelines();
  if (!g_sunMapLow)
    g_sunMapLow = make_sun_map_pipeline(kSunLowFormat, false, true);
  // HD's pre-pass from this frame's depth (the sea mask's, drawn before the scene for this): the map's shadow at
  // half the size, no blend with the last frames'.
  const SunMapInputs inputs = sun_map_inputs(camera, sunMap, true);
  draw_into(cmd, "Characters' shadow (half size)", g_sunMapLow,
            sun_map_group(inputs, g_seaMaskDepthView, g_farView[0], effect_none_view()),
            g_actorLowMip[0]);
  const auto blur_group = [&](const wgpu::TextureView& view) {
    const std::array entries{wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
                             wgpu::BindGroupEntry{.binding = 1, .textureView = view}};
    const wgpu::BindGroupDescriptor descriptor{
        .layout = g_sunBlurLayout, .entryCount = entries.size(), .entries = entries.data()};
    return g_device.CreateBindGroup(&descriptor);
  };
  // Mip 0: HD's 3-tap blur (0 and +-1.3636 texels, 0.294 and 0.353), across then down.
  draw_into(cmd, "Characters' shadow blur (X)", g_sunBlurX, blur_group(g_actorLowMip[0]), g_actorTmpView[0]);
  draw_into(cmd, "Characters' shadow blur (Y)", g_sunBlurY, blur_group(g_actorTmpView[0]), g_actorLowMip[0]);
  // Mip 1: one bilinear tap of mip 0 at an eighth of the scene, the same blur there, and bilinear back up.
  draw_into(cmd, "Characters' shadow mip 1 (down)", g_sunCopy, blur_group(g_actorLowMip[0]), g_actorTmpView[1]);
  draw_into(cmd, "Characters' shadow mip 1 (X)", g_sunBlurX, blur_group(g_actorTmpView[1]), g_actorTmpView[2]);
  draw_into(cmd, "Characters' shadow mip 1 (Y)", g_sunBlurY, blur_group(g_actorTmpView[2]), g_actorTmpView[1]);
  draw_into(cmd, "Characters' shadow mip 1 (up)", g_sunCopy, blur_group(g_actorTmpView[1]), g_actorLowMip[1]);
  g_actorLowReady = true;
  return true;
}

// One pipeline of HD's bloom chain into `format`. compose: screened (or, with debug view 1, replacing) with the
// EFB's alpha kept.
wgpu::RenderPipeline make_bloom_hd_pipeline(const char* label, const char* entry, wgpu::TextureFormat format,
                                            uint32_t samples, bool compose,
                                            const std::vector<wgpu::ConstantEntry>& extra = {}) {
  std::vector<wgpu::ConstantEntry> constants{
      {.key = "intensity", .value = g_bloomHdIntensity},
      {.key = "threshold", .value = g_bloomHdThreshold},
      {.key = "range", .value = g_bloomHdRange},
      {.key = "mask_scale", .value = g_bloomHdScale},
      {.key = "strength", .value = g_bloomHdStrength},
      {.key = "debug_view", .value = static_cast<double>(g_bloomHdDebug)},
  };
  constants.insert(constants.end(), extra.begin(), extra.end());
  const wgpu::BlendState screen{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::OneMinusDst,
                .dstFactor = wgpu::BlendFactor::One},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const wgpu::ColorTargetState target{
      .format = format,
      .blend = compose && g_bloomHdDebug != 1 ? &screen : nullptr,
      .writeMask = compose ? wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue
                           : wgpu::ColorWriteMask::All,
  };
  const wgpu::FragmentState fragment{
      .module = g_bloomHdModule,
      .entryPoint = entry,
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = 1,
      .targets = &target,
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      .layout = g_bloomHdPipelineLayout,
      .vertex = {.module = g_bloomHdModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = compose ? samples : 1u, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

std::vector<wgpu::ConstantEntry> bloom_hd_tints(const float a[3], const float b[3]) {
  return {{.key = "tint_ar", .value = a[0]}, {.key = "tint_ag", .value = a[1]}, {.key = "tint_ab", .value = a[2]},
          {.key = "tint_br", .value = b[0]}, {.key = "tint_bg", .value = b[1]}, {.key = "tint_bb", .value = b[2]}};
}

// HD's bloom chain's module, pipelines and targets for a scene of `size`.
void ensure_bloom_hd(const wgpu::Extent3D& size, wgpu::TextureFormat format, uint32_t samples) {
  if (!g_bloomHdModule) {
    wgpu::ShaderSourceWGSL source{};
    source.code = kBloomHdShader;
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Bloom (HD)"};
    g_bloomHdModule = g_device.CreateShaderModule(&moduleDescriptor);
    const auto texture_entry = [](uint32_t binding) {
      return wgpu::BindGroupLayoutEntry{
          .binding = binding,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D},
      };
    };
    const std::array entries{
        wgpu::BindGroupLayoutEntry{.binding = 0,
                                   .visibility = wgpu::ShaderStage::Fragment,
                                   .sampler = {.type = wgpu::SamplerBindingType::Filtering}},
        texture_entry(1),
        texture_entry(2),
    };
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "Bloom (HD)", .entryCount = entries.size(), .entries = entries.data()};
    g_bloomHdLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
    const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                  .bindGroupLayouts = &g_bloomHdLayout};
    g_bloomHdPipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
    const float white[3] = {1.f, 1.f, 1.f};
    g_bloomHdDetect = make_bloom_hd_pipeline("Bloom (HD) detect", "fs_detect", kSunLowFormat, 1, false);
    g_bloomHdDetectMasked = make_bloom_hd_pipeline("Bloom (HD) detect, masked", "fs_detect", kSunLowFormat, 1, false,
                                                   {{.key = "use_mask", .value = 1.0}});
    g_bloomHdDown = make_bloom_hd_pipeline("Bloom (HD) copy", "fs_down", kSunLowFormat, 1, false);
    g_bloomHdBlurX = make_bloom_hd_pipeline("Bloom (HD) blur (X)", "fs_blur", kSunLowFormat, 1, false,
                                            {{.key = "axis_x", .value = 1.0}});
    g_bloomHdBlurY = make_bloom_hd_pipeline("Bloom (HD) blur (Y)", "fs_blur", kSunLowFormat, 1, false,
                                            {{.key = "axis_x", .value = 0.0}});
    g_bloomHdGather1 = make_bloom_hd_pipeline("Bloom (HD) gather 1/32 into 1/16", "fs_gather", kSunLowFormat, 1,
                                              false, bloom_hd_tints(g_bloomHdColors[1], g_bloomHdColors[2]));
    g_bloomHdGather2 = make_bloom_hd_pipeline("Bloom (HD) gather 1/16 into 1/8", "fs_gather", kSunLowFormat, 1,
                                              false, bloom_hd_tints(g_bloomHdColors[0], white));
  }
  if (!g_bloomHdCompose || format != g_bloomHdFormat || samples != g_bloomHdSamples) {
    g_bloomHdCompose = make_bloom_hd_pipeline("Bloom (HD) compose", "fs_compose", format, samples, true);
    g_bloomHdFormat = format;
    g_bloomHdSamples = samples;
  }
  if (g_bloomHdTex[0] && g_bloomHdSize.width == size.width && g_bloomHdSize.height == size.height)
    return;
  g_bloomHdSize = size;
  for (size_t i = 0; i < g_bloomHdTex.size(); ++i) {
    const uint32_t div = i == 0 ? 4u : i <= 2 ? 8u : i <= 4 ? 16u : 32u;
    const wgpu::TextureDescriptor descriptor{
        .label = "Bloom (HD)",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .size = {std::max(1u, size.width / div), std::max(1u, size.height / div), 1},
        .format = kSunLowFormat,
    };
    g_bloomHdTex[i] = g_device.CreateTexture(&descriptor);
    g_bloomHdView[i] = g_bloomHdTex[i].CreateView();
  }
}

void draw_bloom_hd(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                   const wgpu::TextureView& src, const wgpu::TextureView& src2, const wgpu::TextureView& target,
                   bool load = false, const wgpu::TextureView& resolve = nullptr) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = src},
      wgpu::BindGroupEntry{.binding = 2, .textureView = src2},
  };
  const wgpu::BindGroupDescriptor groupDescriptor{
      .layout = g_bloomHdLayout, .entryCount = entries.size(), .entries = entries.data()};
  const auto group = g_device.CreateBindGroup(&groupDescriptor);
  const wgpu::RenderPassColorAttachment attachment{
      .view = target,
      .resolveTarget = resolve,
      .loadOp = load ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0.0, 0.0, 0.0, 0.0},
  };
  const wgpu::RenderPassDescriptor descriptor{
      .label = label,
      .colorAttachmentCount = 1,
      .colorAttachments = &attachment,
      .timestampWrites = webgpu::gpu_prof::pass_writes(label),
  };
  const auto pass = cmd.BeginRenderPass(&descriptor);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, group, 0, nullptr);
  pass.Draw(3);
  pass.End();
}

void encode_bloom_hd(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::Extent3D& size,
                     wgpu::TextureFormat format, const wgpu::TextureView& dst, const wgpu::TextureView& resolve,
                     uint32_t msaaSamples) {
  const webgpu::gpu_prof::Zone zone{cmd, "Bloom (HD)"};
  ensure_bloom_hd(size, format, msaaSamples);
  const bool masked = bloom_mask_wanted() && g_seaMaskView && g_seaMaskSize.width == size.width &&
                      g_seaMaskSize.height == size.height;
  const wgpu::TextureView& mask = masked ? g_seaMaskView : scene;
  const auto& v = g_bloomHdView;
  draw_bloom_hd(cmd, "Bloom (HD) detect", masked ? g_bloomHdDetectMasked : g_bloomHdDetect, scene, mask, v[0]);
  draw_bloom_hd(cmd, "Bloom (HD) copy 1/8", g_bloomHdDown, v[0], v[0], v[1]);
  draw_bloom_hd(cmd, "Bloom (HD) copy 1/16", g_bloomHdDown, v[0], v[0], v[3]);
  draw_bloom_hd(cmd, "Bloom (HD) copy 1/32", g_bloomHdDown, v[3], v[3], v[5]);
  for (size_t level = 1; level < 7; level += 2) {
    draw_bloom_hd(cmd, "Bloom (HD) blur (X)", g_bloomHdBlurX, v[level], v[level], v[level + 1]);
    draw_bloom_hd(cmd, "Bloom (HD) blur (Y)", g_bloomHdBlurY, v[level + 1], v[level + 1], v[level]);
  }
  draw_bloom_hd(cmd, "Bloom (HD) gather 1/32 into 1/16", g_bloomHdGather1, v[3], v[5], v[4]);
  draw_bloom_hd(cmd, "Bloom (HD) gather 1/16 into 1/8", g_bloomHdGather2, v[1], v[4], v[2]);
  draw_bloom_hd(cmd, "Bloom (HD) compose", g_bloomHdCompose, v[2], g_bloomHdDebug == 2 && masked ? mask : v[2], dst,
                true, resolve);
}
// Wind Waker HD's ww_shadow_clouds from a BC4 DDS (FourCC ATI1 or BC4U, or DX10 with BC4_UNORM), all its mips.
void load_cloud_texture(const char* path) {
  const auto bytes = dds::read_binary_file(path);
  if (!bytes || bytes->size() < 128 || std::memcmp(bytes->data(), "DDS ", 4) != 0) {
    std::fprintf(stderr, "[cloud-shadows] %s: not a DDS\n", path);
    return;
  }
  const uint8_t* d = bytes->data();
  const auto u32at = [&](size_t at) {
    uint32_t v;
    std::memcpy(&v, d + at, 4);
    return v;
  };
  const uint32_t height = u32at(12), width = u32at(16), mips = std::max(u32at(28), 1u), fourcc = u32at(84);
  size_t offset = 128;
  bool bc4 = fourcc == 0x31495441u /* ATI1 */ || fourcc == 0x55344342u /* BC4U */;
  if (fourcc == 0x30315844u /* DX10 */ && bytes->size() >= 148) {
    bc4 = u32at(128) == 80u; // DXGI_FORMAT_BC4_UNORM
    offset = 148;
  }
  if (!bc4 || width == 0 || height == 0) {
    std::fprintf(stderr, "[cloud-shadows] %s: expected a BC4 DDS\n", path);
    return;
  }
  const wgpu::TextureDescriptor descriptor{
      .label = "Cloud shadows (HD)",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = {width, height, 1},
      .format = wgpu::TextureFormat::BC4RUnorm,
      .mipLevelCount = mips,
  };
  g_cloudTex = g_device.CreateTexture(&descriptor);
  for (uint32_t mip = 0; mip < mips; ++mip) {
    const uint32_t w = std::max(width >> mip, 1u), h = std::max(height >> mip, 1u);
    const uint32_t bw = (w + 3) / 4, bh = (h + 3) / 4, row = bw * 8, size = row * bh;
    if (offset + size > bytes->size()) {
      std::fprintf(stderr, "[cloud-shadows] %s: short at mip %u\n", path, mip);
      g_cloudTex = {};
      return;
    }
    const wgpu::TexelCopyTextureInfo dst{.texture = g_cloudTex, .mipLevel = mip};
    const wgpu::TexelCopyBufferLayout layout{.bytesPerRow = row, .rowsPerImage = bh};
    const wgpu::Extent3D extent{bw * 4, bh * 4, 1};
    webgpu::g_queue.WriteTexture(&dst, d + offset, size, &layout, &extent);
    offset += size;
  }
  g_cloudTexView = g_cloudTex.CreateView();
  std::fprintf(stderr, "[cloud-shadows] HD's cloud texture %ux%u, %u mips, from %s\n", width, height, mips, path);
}
} // namespace

bool draw_actor_shadow(const wgpu::CommandEncoder& cmd, const Camera& camera, const SunMap& sunMap,
                       const wgpu::Extent3D& size, wgpu::TextureFormat format) {
  return draw_actor_shadow_now(cmd, camera, sunMap, size, format);
}

void end_actor_shadow() noexcept { g_actorLowReady = false; }

// --- Wind Waker HD's monotone (Frozen Hyrule) -------------------------------------------------------------------
//
// HD's DOF pass while monotone (pixel shader 40716 in the Hyroom capture), on HD's linear scene: a 2x2 box of the
// scene at half its size, its grey (HD's weights) tinted by the colour multiply, laid over the scene by
// clamp(level x clamp((distance - start) / (end - start))); nearer than start it is left out (colour). The frame
// buffer holds what HD's sRGB target ends with, so the pass takes it to linear and back.
namespace {
constexpr const char* kMonotoneShader = R"""(
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var half_scene: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> u: Mono;
@group(0) @binding(4) var scene: texture_2d<f32>;

struct Mono {
    a: vec4<f32>, // near, start, end, level
    b: vec4<f32>, // the colour multiply, saturate
};

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

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, c <= vec3(0.04045));
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    return select(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3(0.0031308));
}

// HD's half-size copy (7019): one bilinear tap at each half-size texel's centre, the 2x2 box under it.
@fragment
fn fs_half(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(textureSampleLevel(half_scene, samp, in.uv, 0.0).rgb, 1.0);
}

@fragment
fn fs_monotone(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy);
    let d = textureLoad(depth, p, 0);
    // No depth (the sky): as far as can be, sepia.
    let dist = select(u.a.x / d, 3.0e38, d <= 0.0);
    // HD's ramp; with start and end the same (MonotoneOut), all of it beyond start.
    var b = select(0.0, 1.0, dist > u.a.y);
    if (u.a.z > u.a.y) {
        b = clamp((dist - u.a.y) / (u.a.z - u.a.y), 0.0, 1.0);
    }
    if (b <= 0.0) {
        discard;
    }
    let a = clamp(u.a.w * b, 0.0, 1.0);
    var c = to_linear(textureSampleLevel(half_scene, samp, in.uv, 0.0).rgb);
    c = mix(c, vec3(dot(c, vec3(0.298912, 0.586611, 0.114478))), u.b.w);
    // HD's target is UNORM: the tinted grey is clamped before it is blended.
    c = clamp(c * u.b.xyz, vec3(0.0), vec3(1.0));
    let s = to_linear(textureLoad(scene, p, 0).rgb);
    return vec4(to_srgb(c * a + s * (1.0 - a)), 1.0);
}

// The EFB's depth at the grey composite, for encode_late().
@fragment
fn fs_depth(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(textureLoad(depth, vec2<i32>(in.pos.xy), 0), 0.0, 0.0, 0.0);
}

// After the player's lists: the lit copy where something was drawn nearer than the snapshot's depth.
@fragment
fn fs_late(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy);
    let d = textureLoad(depth, p, 0);
    let before = textureLoad(scene, p, 0).r;
    // Reversed Z: nearer is larger.
    if (d <= before * 1.0001 + 1.0e-7) {
        discard;
    }
    return vec4(textureLoad(half_scene, p, 0).rgb, 1.0);
}
)""";

std::mutex g_monotoneMutex;
Monotone g_monotone;
wgpu::ShaderModule g_monoModule;
wgpu::BindGroupLayout g_monoLayout; // the far blur composite's, with the monotone's 32-byte uniform
wgpu::PipelineLayout g_monoPipelineLayout;
wgpu::RenderPipeline g_monoHalfPass, g_monoPass, g_monoDepthPass, g_monoLatePass;
wgpu::TextureFormat g_monoFormat = wgpu::TextureFormat::Undefined;
wgpu::Texture g_monoHalf, g_monoCopy, g_monoDepth;
wgpu::TextureView g_monoHalfView, g_monoCopyView, g_monoDepthView;
bool g_monoDepthValid = false;

wgpu::RenderPipeline make_monotone_pipeline(const char* label, const char* entry, wgpu::TextureFormat format,
                                            bool colourOnly) {
  const std::array targets{wgpu::ColorTargetState{
      .format = format,
      .writeMask = colourOnly ? wgpu::ColorWriteMask::Red | wgpu::ColorWriteMask::Green | wgpu::ColorWriteMask::Blue
                              : wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_monoModule,
      .entryPoint = entry,
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = label,
      .layout = g_monoPipelineLayout,
      .vertex = {.module = g_monoModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = 1, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

wgpu::Texture make_monotone_texture(const char* label, uint32_t width, uint32_t height, wgpu::TextureFormat format,
                                    wgpu::TextureUsage usage, wgpu::TextureView& view) {
  const wgpu::TextureDescriptor descriptor{
      .label = label,
      .usage = usage,
      .size = {width, height, 1},
      .format = format,
  };
  auto texture = g_device.CreateTexture(&descriptor);
  view = texture.CreateView();
  return texture;
}

// Its pipelines and targets at the scene's size and format.
void ensure_monotone(const wgpu::Extent3D& size, wgpu::TextureFormat format) {
  if (!g_monoModule) {
    wgpu::ShaderSourceWGSL source{};
    source.code = kMonotoneShader;
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &source, .label = "Monotone (HD)"};
    g_monoModule = g_device.CreateShaderModule(&moduleDescriptor);
    const std::array entries{
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
        wgpu::BindGroupLayoutEntry{
            .binding = 2,
            .visibility = wgpu::ShaderStage::Fragment,
            .texture = {.sampleType = wgpu::TextureSampleType::Depth, .viewDimension = wgpu::TextureViewDimension::e2D},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 3,
            .visibility = wgpu::ShaderStage::Fragment,
            .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = 32},
        },
        wgpu::BindGroupLayoutEntry{
            .binding = 4,
            .visibility = wgpu::ShaderStage::Fragment,
            .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                        .viewDimension = wgpu::TextureViewDimension::e2D},
        },
    };
    const wgpu::BindGroupLayoutDescriptor layoutDescriptor{
        .label = "Monotone (HD)",
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    g_monoLayout = g_device.CreateBindGroupLayout(&layoutDescriptor);
    const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                  .bindGroupLayouts = &g_monoLayout};
    g_monoPipelineLayout = g_device.CreatePipelineLayout(&pipelineLayoutDescriptor);
  }
  if (g_monoFormat != format || !g_monoPass) {
    g_monoHalfPass = make_monotone_pipeline("Monotone half size (HD)", "fs_half", format, false);
    g_monoPass = make_monotone_pipeline("Monotone (HD)", "fs_monotone", format, true);
    g_monoLatePass = make_monotone_pipeline("Monotone, after the player (HD)", "fs_late", format, true);
    g_monoDepthPass =
        make_monotone_pipeline("Monotone depth (HD)", "fs_depth", wgpu::TextureFormat::R32Float, false);
    g_monoFormat = format;
    g_monoHalf = g_monoCopy = {};
  }
  const uint32_t halfW = (size.width + 1) / 2, halfH = (size.height + 1) / 2;
  if (!g_monoHalf || g_monoHalf.GetWidth() != halfW || g_monoHalf.GetHeight() != halfH)
    g_monoHalf = make_monotone_texture("Monotone half size (HD)", halfW, halfH, format,
                                       wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
                                       g_monoHalfView);
  if (!g_monoCopy || g_monoCopy.GetWidth() != size.width || g_monoCopy.GetHeight() != size.height)
    g_monoCopy = make_monotone_texture("Monotone scene copy (HD)", size.width, size.height, format,
                                       wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding |
                                           wgpu::TextureUsage::RenderAttachment,
                                       g_monoCopyView);
  if (!g_monoDepth || g_monoDepth.GetWidth() != size.width || g_monoDepth.GetHeight() != size.height) {
    g_monoDepth = make_monotone_texture("Monotone depth (HD)", size.width, size.height,
                                        wgpu::TextureFormat::R32Float,
                                        wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
                                        g_monoDepthView);
    g_monoDepthValid = false;
  }
}

// The frame buffer into g_monoCopy; false where the target isn't the frame buffer.
bool copy_monotone_scene(const wgpu::CommandEncoder& cmd, const wgpu::Extent3D& size, const wgpu::TextureView& dst) {
  const auto& frame = webgpu::g_frameBuffer;
  if (!frame.texture || dst.Get() != frame.view.Get() || frame.texture.GetWidth() != size.width ||
      frame.texture.GetHeight() != size.height)
    return false;
  const wgpu::TexelCopyTextureInfo from{.texture = frame.texture};
  const wgpu::TexelCopyTextureInfo to{.texture = g_monoCopy};
  const wgpu::Extent3D extent{size.width, size.height, 1};
  cmd.CopyTextureToTexture(&from, &to, &extent);
  return true;
}

void draw_monotone_into(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                        const wgpu::TextureView& target, bool load, const wgpu::TextureView& texture,
                        const wgpu::TextureView& depthView, const wgpu::TextureView& scene, const void* uniform,
                        size_t uniformSize) {
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Monotone (HD)",
      .usage = wgpu::BufferUsage::Uniform,
      .size = uniformSize,
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, uniformSize), uniform, uniformSize);
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = texture},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .buffer = buffer, .offset = 0, .size = uniformSize},
      wgpu::BindGroupEntry{.binding = 4, .textureView = scene},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_monoLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
  const std::array attachments{wgpu::RenderPassColorAttachment{
      .view = target,
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

// The monotone over the frame buffer `dst` (single-sampled), as HD's DOF pass.
void draw_monotone(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
                   const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
                   const Camera& camera, const Monotone& mono) {
  ensure_monotone(size, format);
  if (!copy_monotone_scene(cmd, size, dst))
    return;
  const webgpu::gpu_prof::Zone zone{cmd, "Monotone (HD)"};
  const std::array<float, 8> uniform{camera.view[14], mono.start,  mono.end,    mono.level,
                                     mono.mul[0],     mono.mul[1], mono.mul[2], mono.saturate};
  draw_monotone_into(cmd, "Monotone half size (HD)", g_monoHalfPass, g_monoHalfView, false, scene, depthView,
                     g_monoCopyView, uniform.data(), sizeof uniform);
  draw_monotone_into(cmd, "Monotone (HD)", g_monoPass, dst, true, g_monoHalfView, depthView, g_monoCopyView,
                     uniform.data(), sizeof uniform);
}
} // namespace

void set_monotone(bool on, float level, float start, float end, const float mul[3], float saturate,
                  uint32_t frameCopy) noexcept {
  std::lock_guard lock{g_monotoneMutex};
  g_monotone.on = on;
  g_monotone.level = std::clamp(level, 0.f, 1.f);
  g_monotone.start = start;
  g_monotone.end = end;
  for (int i = 0; i < 3; ++i)
    g_monotone.mul[i] = mul != nullptr ? mul[i] : 1.f;
  g_monotone.saturate = std::clamp(saturate, 0.f, 1.f);
  g_monotone.frameCopy = frameCopy & 0x3FFFFFFFu;
}

Monotone monotone_params() noexcept {
  if (!monotone_hd_supported())
    return {};
  std::lock_guard lock{g_monotoneMutex};
  return g_monotone;
}

bool monotone_hd_supported() noexcept {
  static const bool off = [] {
    const char* value = std::getenv("DOL_AURORA_MONOTONE_HD");
    return value != nullptr && value[0] == '0';
  }();
  return !off && webgpu::g_hasCoreFeatures && webgpu::g_graphicsConfig.msaaSamples <= 1;
}

void snapshot_monotone_depth(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& depthView,
                             const wgpu::Extent3D& size) {
  g_monoDepthValid = false;
  if (!g_module || !depthView || !g_monoModule)
    return;
  ensure_monotone(size, g_monoFormat);
  const std::array<float, 8> none{}; // the shader's Mono, unused
  draw_monotone_into(cmd, "Monotone depth (HD)", g_monoDepthPass, g_monoDepthView, false, g_monoHalfView, depthView,
                     g_monoCopyView, none.data(), sizeof none);
  g_monoDepthValid = true;
}

void encode_late(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
                 const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
                 const wgpu::TextureView& resolve, uint32_t msaaSamples, const Camera& camera,
                 const SunMap* sunMap, bool lateLight) {
  if (lateLight && g_monoDepthValid && g_monoModule && msaaSamples == 1 && depthView && camera.valid &&
      g_monoFormat == format && copy_monotone_scene(cmd, size, dst)) {
    const webgpu::gpu_prof::Zone zone{cmd, "Monotone, after the player (HD)"};
    g_sunTemporalHeld = true;
    encode(cmd, g_monoCopyView, depthView, size, format, g_monoCopyView, nullptr, 1, camera, sunMap,
           Phase::LateLight);
    g_sunTemporalHeld = false;
    const std::array<float, 8> none{}; // the shader's Mono, unused
    draw_monotone_into(cmd, "Monotone, after the player (HD)", g_monoLatePass, dst, true, g_monoCopyView, depthView,
                       g_monoDepthView, none.data(), sizeof none);
  }
  g_monoDepthValid = false;
  if (bloom_enabled())
    encode_bloom(cmd, scene, size, format, dst, resolve, msaaSamples);
}


void initialize() {
  const char* bloom = std::getenv("DOL_AURORA_BLOOM");
  g_bloom.store(bloom != nullptr && bloom[0] == '1', std::memory_order_relaxed);
  g_threshold = env_float("DOL_AURORA_BLOOM_THRESHOLD", g_threshold, 0.f, 1.f);
  g_strength = env_float("DOL_AURORA_BLOOM_STRENGTH", g_strength, 0.f, 4.f);
  if (const char* hd = std::getenv("DOL_AURORA_BLOOM_HD"))
    g_bloomHd = hd[0] == '1';
  if (const char* hdMask = std::getenv("DOL_AURORA_BLOOM_HD_MASK"))
    g_bloomHdMask = hdMask[0] == '1';
  g_bloomHdIntensity = env_float("DOL_AURORA_BLOOM_HD_INTENSITY", g_bloomHdIntensity, 0.f, 8.f);
  g_bloomHdThreshold = env_float("DOL_AURORA_BLOOM_HD_THRESHOLD", g_bloomHdMask ? 0.f : g_threshold, 0.f, 1.f);
  g_bloomHdRange = env_float("DOL_AURORA_BLOOM_HD_RANGE", g_bloomHdMask ? 0.1f : 1.f - g_bloomHdThreshold, 0.001f, 1.f);
  g_bloomHdScale = env_float("DOL_AURORA_BLOOM_HD_SCALE", g_bloomHdScale, 0.f, 8.f);
  g_bloomHdStrength = env_float("DOL_AURORA_BLOOM_HD_STRENGTH", g_bloomHdStrength, 0.f, 8.f);
  for (int i = 0; i < 3; ++i) {
    const std::string name = "DOL_AURORA_BLOOM_HD_COLOR" + std::to_string(i + 1);
    if (const char* colour = std::getenv(name.c_str())) {
      float c[3];
      if (std::sscanf(colour, "%f,%f,%f", &c[0], &c[1], &c[2]) == 3)
        for (int j = 0; j < 3; ++j)
          g_bloomHdColors[i][j] = std::clamp(c[j], 0.f, 4.f);
    }
  }
  if (const char* hdDebug = std::getenv("DOL_AURORA_BLOOM_HD_DEBUG"))
    g_bloomHdDebug = std::atoi(hdDebug);
  const char* far = std::getenv("DOL_AURORA_FAR_BLUR");
  g_farBlur.store(far != nullptr && far[0] == '1', std::memory_order_relaxed);
  g_farStart = env_float("DOL_AURORA_FAR_BLUR_START", g_farStart, 1.f, 1e9f);
  g_farEnd = env_float("DOL_AURORA_FAR_BLUR_END", g_farEnd, g_farStart + 1.f, 1e9f);
  g_farAmount = env_float("DOL_AURORA_FAR_BLUR_AMOUNT", g_farAmount, 0.f, 1.f);
  g_farBackground = env_float("DOL_AURORA_FAR_BLUR_BACKGROUND", g_farBackground, 0.f, 1.f);
  {
    const char* hd = std::getenv("DOL_AURORA_FAR_BLUR_HD");
    g_farHd = hd != nullptr && hd[0] == '1';
  }
  const char* ao = std::getenv("DOL_AURORA_AO");
  g_ao.store(ao != nullptr && ao[0] == '1', std::memory_order_relaxed);
  g_aoRadius = env_float("DOL_AURORA_AO_RADIUS", g_aoRadius, 0.1f, 1e6f);
  g_aoStrength = env_float("DOL_AURORA_AO_STRENGTH", g_aoStrength, 0.f, 1.f);
  g_aoFadeStart = env_float("DOL_AURORA_AO_FADE_START", g_aoFadeStart, 1.f, 1e9f);
  g_aoFadeEnd = env_float("DOL_AURORA_AO_FADE_END", g_aoFadeEnd, g_aoFadeStart + 1.f, 1e9f);
  g_cloudStrength = env_float("DOL_AURORA_CLOUD_SHADOWS_STRENGTH", g_cloudStrength, 0.f, 1.f);
  g_cloudScale = env_float("DOL_AURORA_CLOUD_SHADOWS_SCALE", g_cloudScale, 1.f, 1e7f);
  g_sunStrength = env_float("DOL_AURORA_SUN_SHADOWS_STRENGTH", g_sunStrength, 0.f, 1.f);
  if (const char* colour = std::getenv("DOL_AURORA_SUN_SHADOWS_COLOUR")) {
    float c[3];
    if (std::sscanf(colour, "%f,%f,%f", &c[0], &c[1], &c[2]) == 3)
      for (int i = 0; i < 3; ++i)
        g_sunShadowColour[i] = std::clamp(c[i], 0.f, 1.f);
  }
  g_sunReach = env_float("DOL_AURORA_SUN_SHADOWS_REACH", g_sunReach, 1.f, 1e6f);
  // The host decides through set_sun_shadow_map(); the variable overrides it only when it is set.
  if (const char* sunMap = std::getenv("DOL_AURORA_SUN_SHADOW_MAP"); sunMap != nullptr) {
    g_sunMapOn.store(sunMap[0] == '1', std::memory_order_relaxed);
  }
  // DOL_AURORA_SUN_SHADOW_QUALITY=high (the default), medium or low sets the
  // map's size, its cascades, the filter's lookups and the gap fill together;
  // each of their own variables still wins. The host can change it while
  // running (aurora_set_sun_shadow_quality).
  {
    const char* quality = std::getenv("DOL_AURORA_SUN_SHADOW_QUALITY");
    apply_sun_quality(quality == nullptr                    ? 2
                      : std::strcmp(quality, "low") == 0    ? 0
                      : std::strcmp(quality, "medium") == 0 ? 1
                      : std::strcmp(quality, "hd") == 0     ? 3
                                                            : 2);
  }
  g_sunMapRange = env_float("DOL_AURORA_SUN_SHADOW_MAP_RANGE", g_sunMapRange, 500.f, 100000.f);
  g_sunMapSpread = env_float("DOL_AURORA_SUN_SHADOW_MAP_SPREAD", g_sunMapSpread, 1.5f, 16.f);
  g_sunMapDepth = env_float("DOL_AURORA_SUN_SHADOW_MAP_DEPTH", g_sunMapDepth, 1000.f, 1000000.f);
  g_sunMapBias = env_float("DOL_AURORA_SUN_SHADOW_MAP_BIAS", g_sunMapBias, 0.f, 1000.f);
  g_sunMapSlope = env_float("DOL_AURORA_SUN_SHADOW_MAP_SLOPE", g_sunMapSlope, 0.f, 32.f);
  if (const char* hdBias = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_HD_BIAS"))
    g_sunMapHdBias = hdBias[0] == '1';
  if (const char* hdMix = std::getenv("DOL_AURORA_SUN_SHADOW_HD_MIX"))
    g_sunHdMix = hdMix[0] == '1';
  if (const char* fades = std::getenv("DOL_AURORA_SUN_SHADOW_HD_MIX_FADES"))
    std::sscanf(fades, "%f,%f,%f", &g_sunHdMixFades[0], &g_sunHdMixFades[1], &g_sunHdMixFades[2]);
  g_sunHdMixBlurs = static_cast<int>(env_float("DOL_AURORA_SUN_SHADOW_HD_MIX_BLURS", 1.f, 0.f, 8.f));
  g_sunSeaShadow = env_float("DOL_AURORA_SUN_SHADOW_SEA_STRENGTH", g_sunSeaShadow, 0.f, 1.f);
  g_sunHdMixNearPower = env_float("DOL_AURORA_SUN_SHADOW_HD_MIX_NEAR_POWER", g_sunHdMixNearPower, 1.f, 8.f);
  if (const char* skinned = std::getenv("DOL_AURORA_SUN_SHADOW_SKINNED_SCENERY"))
    g_sunSkinnedScenery = skinned[0] == '1';
  if (const char* temporal = std::getenv("DOL_AURORA_SUN_SHADOW_TEMPORAL"))
    g_sunTemporal = temporal[0] == '1';
  if (const char* range = std::getenv("DOL_AURORA_SUN_SHADOW_TEMPORAL_RANGE"))
    std::sscanf(range, "%f,%f,%f", &g_sunTemporalRange[0], &g_sunTemporalRange[1], &g_sunTemporalRange[2]);
  g_sunTemporalRange[2] = std::clamp(g_sunTemporalRange[2], 0.f, 0.98f);
  if (const char* aoHd = std::getenv("DOL_AURORA_AO_HD"))
    g_aoHd = aoHd[0] == '1';
  g_aoHdRadius = env_float("DOL_AURORA_AO_HD_RADIUS", g_aoHdRadius, 0.f, 1.f);
  g_aoHdFar = env_float("DOL_AURORA_AO_HD_FAR", g_aoHdFar, 1.f, 1e7f);
  g_aoHdAttn = env_float("DOL_AURORA_AO_HD_ATTN", g_aoHdAttn, 1.f, 1e7f);
  g_aoHdDensity = env_float("DOL_AURORA_AO_HD_DENSITY", g_aoHdDensity, 0.f, 10.f);
  g_aoHdOffset = env_float("DOL_AURORA_AO_HD_OFFSET", g_aoHdOffset, 0.f, 0.5f);
  g_aoHdPower = env_float("DOL_AURORA_AO_HD_POWER", g_aoHdPower, 0.1f, 10.f);
  if (const char* aoFacing = std::getenv("DOL_AURORA_AO_HD_FACING"))
    g_aoHdFacing = aoFacing[0] != '0';
  if (const char* aoDebug = std::getenv("DOL_AURORA_AO_HD_DEBUG"))
    g_aoHdDebug = aoDebug[0] == '1';
  if (const char* cloudHd = std::getenv("DOL_AURORA_CLOUD_SHADOWS_HD"))
    g_cloudHd = cloudHd[0] == '1';
  g_cloudHdPower = env_float("DOL_AURORA_CLOUD_SHADOWS_HD_POWER", g_cloudHdPower, 0.f, 1.f);
  g_cloudHdSpeed = env_float("DOL_AURORA_CLOUD_SHADOWS_HD_SPEED", g_cloudHdSpeed, -10.f, 10.f);
  g_cloudHdRepeat = env_float("DOL_AURORA_CLOUD_SHADOWS_HD_REPEAT", g_cloudHdRepeat, 1.f, 1e7f);
  if (const char* rimHd = std::getenv("DOL_AURORA_RIM_HD"))
    g_rimHd = rimHd[0] == '1';
  g_rimHdScale = env_float("DOL_AURORA_RIM_HD_SCALE", g_rimHdScale, 0.f, 10.f);
  g_rimHdIntensity = env_float("DOL_AURORA_RIM_HD_INTENSITY", g_rimHdIntensity, 0.f, 10.f);
  g_rimHdSharpness = env_float("DOL_AURORA_RIM_HD_SHARPNESS", g_rimHdSharpness, 0.01f, 10.f);
  g_rimHdRim = env_float("DOL_AURORA_RIM_HD_RIM", g_rimHdRim, 0.f, 1.f);
  if (const char* rimDebug = std::getenv("DOL_AURORA_RIM_HD_DEBUG"))
    g_rimHdDebug = std::atoi(rimDebug);
  g_sunMapFadeLow = env_float("DOL_AURORA_SUN_SHADOW_MAP_FADE_LOW", g_sunMapFadeLow, -0.5f, 0.9f);
  g_sunMapFadeHigh = env_float("DOL_AURORA_SUN_SHADOW_MAP_FADE_HIGH", g_sunMapFadeHigh, g_sunMapFadeLow + 0.001f, 1.f);
  g_sunMapHold = env_float("DOL_AURORA_SUN_SHADOW_MAP_HOLD", g_sunMapHold, 0.f, 0.5f);
  g_sunMapDim = env_float("DOL_AURORA_SUN_SHADOW_MAP_DIM", g_sunMapDim, 0.f, 1.f);
  if (const char* facing = std::getenv("DOL_AURORA_SUN_SHADOW_FACING"))
    g_sunFacing = facing[0] == '1';
  if (const char* fog = std::getenv("DOL_AURORA_SUN_FOG_HD"))
    g_sunFogHd = fog[0] != '0';
  g_sunFacingMul = env_float("DOL_AURORA_SUN_SHADOW_FACING_MUL", g_sunFacingMul, 0.5f, 1000.f);
  g_sunFacingDepthMul = env_float("DOL_AURORA_SUN_SHADOW_FACING_DEPTH_MUL", g_sunFacingDepthMul, 0.f, 1000.f);
  g_sunFacingDepthBlur = env_float("DOL_AURORA_SUN_SHADOW_FACING_DEPTH_BLUR", g_sunFacingDepthBlur, 0.f, 200.f);
  if (const char* ambient = std::getenv("DOL_AURORA_SUN_SHADOW_SMOOTH_AMBIENT"))
    g_sunSmoothAmbient = ambient[0] == '1' || ambient[0] == 's', g_sunSmoothAmbientSkinned = ambient[0] == 's';
  if (const char* prepassFar = std::getenv("DOL_AURORA_SUN_SHADOW_PREPASS_FAR"))
    g_sunPrepassFar = prepassFar[0] == '1';
  g_sunFacingFarStart = env_float("DOL_AURORA_SUN_SHADOW_FACING_FAR_START", g_sunFacingFarStart, 0.f, 1e7f);
  g_sunFacingFarEnd = env_float("DOL_AURORA_SUN_SHADOW_FACING_FAR_END", g_sunFacingFarEnd, g_sunFacingFarStart + 1.f, 1e7f);
  if (const char* shape = std::getenv("DOL_AURORA_SUN_SHADOW_SHAPE"))
    g_sunShape = shape[0] != '0';
  g_sunShapeNear = env_float("DOL_AURORA_SUN_SHADOW_SHAPE_NEAR", g_sunShapeNear, 0.f, 1e7f);
  g_sunShapeFar = env_float("DOL_AURORA_SUN_SHADOW_SHAPE_FAR", g_sunShapeFar, g_sunShapeNear + 1.f, 1e7f);
  g_moonStrength = env_float("DOL_AURORA_MOON_SHADOWS_STRENGTH", g_moonStrength, 0.f, 1.f);
  g_moonFadeLow = env_float("DOL_AURORA_MOON_SHADOWS_FADE_LOW", g_moonFadeLow, 0.f, 0.9f);
  g_moonFadeHigh = env_float("DOL_AURORA_MOON_SHADOWS_FADE_HIGH", g_moonFadeHigh, g_moonFadeLow + 0.001f, 1.f);
  g_moonDawnLow = env_float("DOL_AURORA_MOON_SHADOWS_DAWN_LOW", g_moonDawnLow, 0.f, 0.9f);
  g_moonDawnHigh = env_float("DOL_AURORA_MOON_SHADOWS_DAWN_HIGH", g_moonDawnHigh, g_moonDawnLow + 0.001f, 1.f);
  g_sunMapFarStart = env_float("DOL_AURORA_SUN_SHADOW_MAP_FAR_START", g_sunMapFarStart, 0.f, 100000.f);
  g_sunMapFarEnd = env_float("DOL_AURORA_SUN_SHADOW_MAP_FAR_END", g_sunMapFarEnd, g_sunMapFarStart + 1.f, 200000.f);
  g_sunMapNormalNear = env_float("DOL_AURORA_SUN_SHADOW_MAP_NORMAL_NEAR", g_sunMapNormalNear, 0.f, 16.f);
  g_sunMapSlopeNear = env_float("DOL_AURORA_SUN_SHADOW_MAP_SLOPE_NEAR", g_sunMapSlopeNear, 0.f, 32.f);
  g_sunMapNormalOffset = env_float("DOL_AURORA_SUN_SHADOW_MAP_NORMAL_OFFSET", g_sunMapNormalOffset, 0.f, 16.f);
  g_sunMapContact = env_float("DOL_AURORA_SUN_SHADOW_MAP_CONTACT", g_sunMapContact, 20.f, 5000.f);
  g_effectDebug = std::getenv("DOL_AURORA_EFFECT_MASK_DEBUG") != nullptr;
  g_effectGain = env_float("DOL_AURORA_EFFECT_MASK_GAIN", 2.f, 0.f, 1000.f);
  g_effectPush = env_float("DOL_AURORA_EFFECT_MASK_PUSH", 0.05f, 0.f, 0.9f);
  const char* sunMapDebug = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_DEBUG");
  g_sunMapDebug = sunMapDebug != nullptr ? std::atoi(sunMapDebug) : 0;
  const char* sunMapSteps = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_STEPS");
  g_sunMapSteps = sunMapSteps != nullptr && sunMapSteps[0] == '1';
  g_sunMapSeaLow = env_float("DOL_AURORA_SUN_SHADOW_MAP_SEA_LOW", g_sunMapSeaLow, 0.f, 1000.f);
  g_sunMapSeaRise = env_float("DOL_AURORA_SUN_SHADOW_MAP_SEA_RISE", g_sunMapSeaRise, 0.f, 1000.f);
  g_sunMapSeaBias = env_float("DOL_AURORA_SUN_SHADOW_MAP_SEA_BIAS", g_sunMapSeaBias, 0.f, 1.f);
  g_sunMapSoft = env_float("DOL_AURORA_SUN_SHADOW_MAP_SOFT", g_sunMapSoft, 0.f, 1.f);
  g_sunMapSoftMax = env_float("DOL_AURORA_SUN_SHADOW_MAP_SOFT_MAX", g_sunMapSoftMax, 0.f, 16.f);
  if (const char* sunMapSmooth = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_SMOOTH"); sunMapSmooth != nullptr)
    g_sunMapSmooth = sunMapSmooth[0] != '0';
  g_edgeStrength = env_float("DOL_AURORA_EDGE_LIGHT_STRENGTH", g_edgeStrength, 0.f, 4.f);
  g_torchStrength = env_float("DOL_AURORA_TORCH_LIGHT_STRENGTH", g_torchStrength, 0.f, 4.f);
  g_torchRadiusScale = env_float("DOL_AURORA_TORCH_LIGHT_REACH", g_torchRadiusScale, 0.1f, 50.f);
  const char* cloudDebug = std::getenv("DOL_AURORA_CLOUD_SHADOWS_DEBUG");
  g_cloudDebug = cloudDebug != nullptr && cloudDebug[0] == '1';
  const char* aoDebug = std::getenv("DOL_AURORA_AO_DEBUG");
  g_aoDebug = aoDebug != nullptr && (aoDebug[0] == '1' || aoDebug[0] == '2') ? aoDebug[0] - '0' : 0;
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
  wgpu::ShaderSourceWGSL aoSource{};
  aoSource.code = kAoShader;
  const wgpu::ShaderModuleDescriptor aoModuleDescriptor{.nextInChain = &aoSource, .label = "Ambient occlusion"};
  g_aoModule = g_device.CreateShaderModule(&aoModuleDescriptor);
  wgpu::ShaderSourceWGSL cloudSource{};
  const std::string cloudCode = std::string(kCloudNoise) + kCloudShader;
  cloudSource.code = cloudCode.c_str();
  const wgpu::ShaderModuleDescriptor cloudModuleDescriptor{.nextInChain = &cloudSource, .label = "Cloud shadows"};
  g_cloudModule = g_device.CreateShaderModule(&cloudModuleDescriptor);
  wgpu::ShaderSourceWGSL sunSource{};
  sunSource.code = kSunShader;
  const wgpu::ShaderModuleDescriptor sunModuleDescriptor{.nextInChain = &sunSource, .label = "Sun shadows"};
  g_sunModule = g_device.CreateShaderModule(&sunModuleDescriptor);
  wgpu::ShaderSourceWGSL edgeSource{};
  edgeSource.code = kEdgeShader;
  const wgpu::ShaderModuleDescriptor edgeModuleDescriptor{.nextInChain = &edgeSource, .label = "Edge light"};
  g_edgeModule = g_device.CreateShaderModule(&edgeModuleDescriptor);
  wgpu::ShaderSourceWGSL torchSource{};
  torchSource.code = kTorchShader;
  const wgpu::ShaderModuleDescriptor torchModuleDescriptor{.nextInChain = &torchSource, .label = "Torch light"};
  g_torchModule = g_device.CreateShaderModule(&torchModuleDescriptor);
  const std::array depthEntries{
      layoutEntries[0],
      layoutEntries[1],
      wgpu::BindGroupLayoutEntry{
          .binding = 2,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Depth, .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      // The effects' mask (effect_mask_target(), read with textureLoad), which the occlusion leaves.
      wgpu::BindGroupLayoutEntry{
          .binding = 3,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
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
  const std::array farEntries{
      depthEntries[0],
      depthEntries[1],
      depthEntries[2],
      wgpu::BindGroupLayoutEntry{
          .binding = 3,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = 16},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 4,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor farLayoutDescriptor{
      .label = "Far blur composite",
      .entryCount = farEntries.size(),
      .entries = farEntries.data(),
  };
  g_farLayout = g_device.CreateBindGroupLayout(&farLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor farPipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                   .bindGroupLayouts = &g_farLayout};
  g_farPipelineLayout = g_device.CreatePipelineLayout(&farPipelineLayoutDescriptor);
  // Binding 1 is the exposure's sea mask, or for the cloud shadows and the
  // screen's sun shadows the effects' mask (both read with textureLoad); the
  // edge light leaves it.
  const std::array cloudEntries{
      depthEntries[0],
      wgpu::BindGroupLayoutEntry{
          .binding = 1,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      depthEntries[2],
      wgpu::BindGroupLayoutEntry{
          .binding = 3,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(g_camera)},
      },
      // The scene before the effects, for the exposure's highlights (read with textureLoad).
      wgpu::BindGroupLayoutEntry{
          .binding = 4,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      },
  };
  const wgpu::BindGroupLayoutDescriptor cloudLayoutDescriptor{
      .label = "Cloud shadows",
      .entryCount = cloudEntries.size(),
      .entries = cloudEntries.data(),
  };
  g_cloudLayout = g_device.CreateBindGroupLayout(&cloudLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor cloudPipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                     .bindGroupLayouts = &g_cloudLayout};
  g_cloudPipelineLayout = g_device.CreatePipelineLayout(&cloudPipelineLayoutDescriptor);
  auto torchEntries = cloudEntries;
  torchEntries[3].buffer.minBindingSize = kTorchUniformSize;
  const wgpu::BindGroupLayoutDescriptor torchLayoutDescriptor{
      .label = "Torch light",
      .entryCount = torchEntries.size(),
      .entries = torchEntries.data(),
  };
  g_torchLayout = g_device.CreateBindGroupLayout(&torchLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor torchPipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                     .bindGroupLayouts = &g_torchLayout};
  g_torchPipelineLayout = g_device.CreatePipelineLayout(&torchPipelineLayoutDescriptor);

  wgpu::ShaderSourceWGSL sunMapSource{};
  const bool sunFog = g_sunFogHd && webgpu::g_dualSourceBlendingSupported;
  const std::string sunMapCode = std::string(sunFog ? "enable dual_source_blending;\n" : "") + kCloudNoise +
                                 kSunMapShader + (sunFog ? kSunMapFogEntry : "");
  sunMapSource.code = sunMapCode.c_str();
  const wgpu::ShaderModuleDescriptor sunMapModuleDescriptor{.nextInChain = &sunMapSource,
                                                            .label = "Sun shadows (shadow map)"};
  g_sunMapModule = g_device.CreateShaderModule(&sunMapModuleDescriptor);
  auto normalEntry = cloudEntries[1];
  normalEntry.binding = 6;
  auto blurEntry = cloudEntries[1];
  blurEntry.binding = 7;
  const wgpu::BindGroupLayoutEntry cloudMapEntry{
      .binding = 8,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = {.sampleType = wgpu::TextureSampleType::Float, .viewDimension = wgpu::TextureViewDimension::e2D},
  };
  const wgpu::BindGroupLayoutEntry cloudSamplerEntry{
      .binding = 9,
      .visibility = wgpu::ShaderStage::Fragment,
      .sampler = {.type = wgpu::SamplerBindingType::Filtering},
  };
  std::array<wgpu::BindGroupLayoutEntry, 11> sunMapEntries{
      cloudEntries[0],
      cloudEntries[1],
      cloudEntries[2],
      cloudEntries[3],
      normalEntry,
      blurEntry,
      cloudMapEntry,
      cloudSamplerEntry,
      // The effects' mask (effect_mask_target(), read with textureLoad), which the shade leaves.
      wgpu::BindGroupLayoutEntry{
          .binding = 10,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                      .viewDimension = wgpu::TextureViewDimension::e2D},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 4,
          .visibility = wgpu::ShaderStage::Fragment,
          .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                      .viewDimension = wgpu::TextureViewDimension::e2DArray},
      },
      wgpu::BindGroupLayoutEntry{
          .binding = 5,
          .visibility = wgpu::ShaderStage::Fragment,
          .sampler = {.type = wgpu::SamplerBindingType::Comparison},
      },
  };
  sunMapEntries[3].buffer.minBindingSize = kSunMapUniformFloats * sizeof(float);
  const wgpu::BindGroupLayoutDescriptor sunMapLayoutDescriptor{
      .label = "Sun shadows (shadow map)",
      .entryCount = sunMapEntries.size(),
      .entries = sunMapEntries.data(),
  };
  g_sunMapLayout = g_device.CreateBindGroupLayout(&sunMapLayoutDescriptor);
  const wgpu::PipelineLayoutDescriptor sunMapPipelineLayoutDescriptor{.bindGroupLayoutCount = 1,
                                                                      .bindGroupLayouts = &g_sunMapLayout};
  g_sunMapPipelineLayout = g_device.CreatePipelineLayout(&sunMapPipelineLayoutDescriptor);
  const wgpu::SamplerDescriptor compareDescriptor{
      .label = "Sun shadow map",
      .addressModeU = wgpu::AddressMode::ClampToEdge,
      .addressModeV = wgpu::AddressMode::ClampToEdge,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .compare = wgpu::CompareFunction::LessEqual,
  };
  g_sunMapCompare = g_device.CreateSampler(&compareDescriptor);
  const wgpu::SamplerDescriptor cloudSamplerDescriptor{
      .label = "Cloud shadows (HD)",
      .addressModeU = wgpu::AddressMode::Repeat,
      .addressModeV = wgpu::AddressMode::Repeat,
      .magFilter = wgpu::FilterMode::Linear,
      .minFilter = wgpu::FilterMode::Linear,
      .mipmapFilter = wgpu::MipmapFilterMode::Linear,
  };
  g_cloudSampler = g_device.CreateSampler(&cloudSamplerDescriptor);
  if (const char* path = std::getenv("DOL_AURORA_CLOUD_SHADOWS_HD_TEXTURE"); path != nullptr && path[0] != '\0')
    load_cloud_texture(path);
}

void shutdown() {
  g_monoHalfPass = g_monoPass = g_monoDepthPass = g_monoLatePass = {};
  g_monoHalf = g_monoCopy = g_monoDepth = {};
  g_monoHalfView = g_monoCopyView = g_monoDepthView = {};
  g_monoModule = {};
  g_monoPipelineLayout = {};
  g_monoLayout = {};
  g_monoFormat = wgpu::TextureFormat::Undefined;
  g_monoDepthValid = false;
  g_bloomHdDetect = g_bloomHdDetectMasked = g_bloomHdDown = g_bloomHdBlurX = g_bloomHdBlurY = {};
  g_bloomHdGather1 = g_bloomHdGather2 = g_bloomHdCompose = {};
  g_bloomHdTex = {};
  g_bloomHdView = {};
  g_bloomHdSize = {};
  g_bloomHdFormat = wgpu::TextureFormat::Undefined;
  g_bloomHdSamples = 0;
  g_bloomHdPipelineLayout = {};
  g_bloomHdLayout = {};
  g_bloomHdModule = {};
  g_rimPass = {};
  g_rimSamples = 0;
  g_rimModule = {};
  g_bright = g_blurH = g_blurV = g_composite = g_down = g_farComposite = g_aoPass = g_aoApply = g_exposureDepth = {};
  g_aoBlurH = g_aoBlurV = g_aoUpscale = {};
  g_aoTex = {};
  g_aoView = {};
  g_aoSamples = 0;
  g_aoModule = {};
  g_cloudPass = {};
  g_cloudSamples = 0;
  g_sunPass = {};
  g_sunSamples = 0;
  g_sunModule = {};
  g_sunMapPass = {};
  g_sunMapDirect = {};
  g_sunMapLow = {};
  g_sunCopy = {};
  g_actorLowView = {};
  g_actorLow = {};
  for (int i = 0; i < 3; ++i) {
    g_actorTmpView[i] = {};
    g_actorTmp[i] = {};
  }
  g_actorLowSize = {};
  g_actorLowReady = false;
  g_sunBlurX = g_sunBlurY = {};
  g_sunBlurPipelineLayout = {};
  g_sunBlurLayout = {};
  g_sunBlurModule = {};
  for (int i = 0; i < 2; ++i) {
    g_sunLowView[i] = {};
    g_actorLowMip[i] = {};
    g_sunLowMipView[i][0] = g_sunLowMipView[i][1] = {};
    g_sunLow[i] = {};
  }
  g_sunHistView = {};
  g_sunHist = {};
  g_sunHistValid = false;
  g_sunTemporalPipeline = {};
  g_sunTemporalPipelineLayout = {};
  g_sunTemporalLayout = {};
  g_sunTemporalModule = {};
  g_sunLowSize = {};
  g_sunMapModule = {};
  g_sunMapPipelineLayout = {};
  g_sunMapLayout = {};
  g_sunMapCompare = {};
  g_cloudSampler = {};
  g_cloudTexView = {};
  g_cloudTex = {};
  g_sunMapView = {};
  g_sunMapLayers = {};
  g_sunMapTex = {};
  g_edgePass = {};
  g_edgeSamples = 0;
  g_edgeModule = {};
  g_torchPass = {};
  g_torchSamples = 0;
  g_torchPipelineLayout = {};
  g_torchLayout = {};
  g_torchModule = {};
  g_cloudPipelineLayout = {};
  g_cloudLayout = {};
  g_cloudModule = {};
  g_seaMaskView = {};
  g_seaMask = {};
  g_exposureSceneView = {};
  g_exposureScene = {};
  g_seaMaskDepthView = {};
  g_seaMaskDepth = {};
  g_seaMaskSize = {};
  g_effectMaskView = {};
  g_effectMask = {};
  g_effectMaskDrawn = false;
  g_effectNoneView = {};
  g_effectNone = {};
  g_sunNormalView = {};
  g_sunNormal = {};
  g_sunNormalSize = {};
  g_tex = {};
  g_view = {};
  g_farTex = {};
  g_farView = {};
  g_farSamples = 0;
  g_depthPipelineLayout = {};
  g_depthLayout = {};
  g_farPipelineLayout = {};
  g_farLayout = {};
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

void set_cloud_hd_floor(float floor) noexcept {
  g_cloudHdFloor.store(floor >= 0.f ? std::min(floor, 1.f) : -1.f, std::memory_order_relaxed);
}

namespace {
std::atomic<bool> g_sunCasterCull{false};
} // namespace

void set_sun_caster_cull(bool on) noexcept { g_sunCasterCull.store(on, std::memory_order_relaxed); }

bool sun_caster_cull() noexcept {
  static const int forced = [] {
    const char* value = std::getenv("DOL_AURORA_SUN_SHADOW_CASTER_CULL");
    return value == nullptr || value[0] == '\0' ? -1 : value[0] == '1' ? 1 : 0;
  }();
  return forced >= 0 ? forced == 1 : g_sunCasterCull.load(std::memory_order_relaxed);
}

void set_far_blur_level(float level) noexcept {
  g_farLevel.store(std::clamp(level, 0.f, 4.f), std::memory_order_relaxed);
}

bool ambient_occlusion_enabled() noexcept { return g_ao.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_ambient_occlusion(bool enabled) noexcept { g_ao.store(enabled, std::memory_order_relaxed); }

bool cloud_shadows_enabled() noexcept { return g_clouds.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_cloud_shadows(bool enabled) noexcept { g_clouds.store(enabled, std::memory_order_relaxed); }

bool sun_shadows_enabled() noexcept { return g_sun.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_sun_shadows(bool enabled) noexcept { g_sun.store(enabled, std::memory_order_relaxed); }

bool sun_shadow_map_enabled() noexcept { return sun_shadows_enabled() && g_sunMapOn.load(std::memory_order_relaxed); }

void set_sun_shadow_map(bool enabled) noexcept { g_sunMapOn.store(enabled, std::memory_order_relaxed); }

void set_sun_shadow_quality(int level) noexcept {
  if (level >= 0 && level <= 3)
    g_sunQualityPending.store(level, std::memory_order_relaxed);
}

void set_sun_shadow_strength(float strength) noexcept {
  g_sunMapStrength.store(std::clamp(strength, 0.f, 1.f), std::memory_order_relaxed);
}

void set_sun_cascade_splits(float middle, float far) noexcept {
  const bool ok = std::isfinite(middle) && std::isfinite(far) && middle > 0.f && far > middle;
  g_sunHdLiveSplits[0].store(ok ? middle : 0.f, std::memory_order_relaxed);
  g_sunHdLiveSplits[1].store(ok ? far : 0.f, std::memory_order_relaxed);
}

void set_hd_cloud_texture(const char* path) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_cloudTexPath = path != nullptr ? path : "";
  g_cloudTexPending = true;
}

void set_sun_hd_light(bool on, bool night, float fade) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_sunHdLight = {.on = on, .night = night, .fade = std::clamp(fade, 0.f, 1.f)};
}

void set_sun_palette(bool on, const float sky[3], const float ground[3], const float lit[3]) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_sunPalette.on = on && sky != nullptr && ground != nullptr && lit != nullptr;
  for (int i = 0; g_sunPalette.on && i < 3; ++i) {
    g_sunPalette.sky[i] = std::max(sky[i], 0.f);
    g_sunPalette.ground[i] = std::max(ground[i], 0.f);
    g_sunPalette.lit[i] = std::max(lit[i], 0.f);
  }
}

bool fog_hd() noexcept { return g_fogHd.load(std::memory_order_relaxed); }

void set_fog_hd(bool on) noexcept { g_fogHd.store(on, std::memory_order_relaxed); }

void set_fog_hd_starts(const float* start_end, int count) noexcept {
  std::lock_guard lock{g_sceneFogMutex};
  g_fogHdStarts.fill(0.f);
  for (int i = 0; start_end != nullptr && i < std::min(count, 8); ++i) {
    g_fogHdStarts[size_t(2 * i)] = start_end[2 * i];
    g_fogHdStarts[size_t(2 * i + 1)] = start_end[2 * i + 1];
  }
}

void note_scene_fog(float a, int b_magnitude, int b_shift, float c, const int colour[3], bool hd_square,
                    const float* range, float centre, float width) noexcept {
  std::lock_guard lock{g_sceneFogMutex};
  g_sceneFog.range = range != nullptr && width > 0.f;
  for (int i = 0; i < 12; ++i)
    g_sceneFog.table[i] = g_sceneFog.range ? range[i] : 0.f;
  g_sceneFog.centre = centre;
  g_sceneFog.width = width;
  g_sceneFog.on = true;
  g_sceneFog.stale = 0;
  g_sceneFog.a = a;
  g_sceneFog.b = static_cast<float>(b_magnitude);
  g_sceneFog.shift = static_cast<float>(b_shift);
  g_sceneFog.c = c;
  for (int i = 0; i < 3; ++i)
    g_sceneFog.colour[i] = static_cast<float>(std::clamp(colour[i], 0, 255)) / 255.f;
  g_sceneFog.square = hd_square;
}

bool actor_light_hd_switch() noexcept {
  static const bool on = [] {
    const char* env = std::getenv("DOL_AURORA_ACTOR_LIGHT_HD");
    return env != nullptr && env[0] == '1';
  }();
  return on;
}

void set_actor_light_hd(bool on, const float c0[3], const float shadow[3], const float k[3]) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_actorLight.on = on && c0 != nullptr && shadow != nullptr && k != nullptr;
  for (int i = 0; g_actorLight.on && i < 3; ++i) {
    g_actorLight.c0[i] = std::max(c0[i], 0.f);
    g_actorLight.shadow[i] = std::max(shadow[i], 0.f);
    g_actorLight.k[i] = std::max(k[i], 0.f);
  }
}

void set_actor_light_hd_own(bool on, const float c0[3], const float k[3]) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_actorLightOwn.on = on && c0 != nullptr && k != nullptr;
  for (int i = 0; g_actorLightOwn.on && i < 3; ++i) {
    g_actorLightOwn.c0[i] = std::max(c0[i], 0.f);
    g_actorLightOwn.k[i] = std::max(k[i], 0.f);
  }
}

// DOL_AURORA_ACTOR_LIGHT_HD_OWN=0 (to compare): every character on the host's colours (Link's floor's).
bool actor_light_own(float c0[3], float k[3]) noexcept {
  static const bool allowed = [] {
    const char* env = std::getenv("DOL_AURORA_ACTOR_LIGHT_HD_OWN");
    return env == nullptr || env[0] != '0';
  }();
  std::lock_guard lock{g_cameraMutex};
  for (int i = 0; i < 3; ++i) {
    c0[i] = g_actorLightOwn.c0[i];
    k[i] = g_actorLightOwn.k[i];
  }
  return allowed && g_actorLightOwn.on;
}

bool actor_light_palette(float c0[3], float shadow[3], float k[3]) noexcept {
  std::lock_guard lock{g_cameraMutex};
  for (int i = 0; i < 3; ++i) {
    c0[i] = g_actorLight.c0[i];
    shadow[i] = g_actorLight.shadow[i];
    k[i] = g_actorLight.k[i];
  }
  return g_actorLight.on;
}

bool point_lights_hd_switch() noexcept {
  static const bool on = [] {
    const char* env = std::getenv("DOL_AURORA_POINT_LIGHTS_HD");
    return env != nullptr && env[0] == '1';
  }();
  return on;
}

bool point_lights_hd_active() noexcept { return point_lights_hd_switch() && torch_light_enabled() && actor_light_wanted(); }

bool actor_light_wanted() noexcept {
  if (!actor_light_hd_switch())
    return false;
  std::lock_guard lock{g_cameraMutex};
  return g_actorLight.on;
}

bool actor_light_same_frame() noexcept {
  static const bool on = [] {
    const char* env = std::getenv("DOL_AURORA_ACTOR_LIGHT_SAME_FRAME");
    return env == nullptr || env[0] != '0';
  }();
  return on;
}

bool actor_shadow_source(wgpu::TextureView& view, float& lod, bool& sameFrame) noexcept {
  sameFrame = false;
  if (g_actorLowReady && g_actorLowView) {
    view = g_actorLowView;
    lod = 1.f; // HD's mip 1
    sameFrame = true;
    return true;
  }
  if (!g_sunLow[0] || g_postEncodes == 0 || g_sunLowEncode != g_postEncodes)
    return false;
  view = g_sunLowView[0];
  lod = g_sunHdPrepass ? 1.f : 0.f;
  return true;
}

bool actor_cloud_source(const Camera& camera, float cloud[4], wgpu::TextureView& view, wgpu::Sampler& sampler) noexcept {
  if (!clouds_hd_on() || !g_cloudTexView || !g_cloudSampler)
    return false;
  // fs_sun_map's uv: (-x / cloud_repeat + time x cloud_speed, -z / cloud_repeat + 0.5). u0 is taken modulo 1 (the
  // sampler repeats) so a long session's seconds keep their precision.
  const double u0 = static_cast<double>(camera.view[16]) * static_cast<double>(g_cloudHdSpeed);
  cloud[0] = static_cast<float>(u0 - std::floor(u0));
  cloud[1] = 0.5f;
  cloud[2] = 1.f / g_cloudHdRepeat;
  cloud[3] = cloud_hd_floor(camera);
  view = g_cloudTexView;
  sampler = g_cloudSampler;
  return true;
}

void set_actor_light_applied(bool applied) noexcept { g_actorLightApplied.store(applied, std::memory_order_relaxed); }

bool actor_light_applied() noexcept { return g_actorLightApplied.load(std::memory_order_relaxed); }

// A cascade's half size held in steps of a quarter of a power of two, rounded up: it grows as soon as the slice
// needs more and shrinks only once the slice fits in two steps less, so a camera that sways keeps its texels.
float held_size(float want, float& held) noexcept {
  constexpr float kStep = 0.25f;
  if (held > 0.f && want <= held && want >= held * 0.70710678f)
    return held;
  held = std::exp2(std::ceil(std::log2(std::max(want, 1.f)) / kStep) * kStep);
  return held;
}

// The cascades fitted to the view's slices, turned with the camera (g_follow, sun_map()): x0, y0 the frame fixed to
// the sun, from which the camera's is turned.
void follow_cascades(SunMap& map, const Camera& camera, const float sun[3], const float x0[3], const float y0[3]) {
  const float* v = camera.view.data();
  FollowCascades& f = g_followState;
  const auto dot3 = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  // The camera's right and its forward turned a quarter about the sun, both flat to the sun: their sum points
  // along the camera's right on the map whichever of them lines up with the sun (the inverse view's columns).
  const float right[3] = {v[0], v[4], v[8]}, forward[3] = {-v[2], -v[6], -v[10]};
  float rs = dot3(right, sun), fs = dot3(forward, sun);
  const float rp[3] = {right[0] - rs * sun[0], right[1] - rs * sun[1], right[2] - rs * sun[2]};
  const float fp[3] = {forward[0] - fs * sun[0], forward[1] - fs * sun[1], forward[2] - fs * sun[2]};
  const float fx[3] = {fp[1] * sun[2] - fp[2] * sun[1], fp[2] * sun[0] - fp[0] * sun[2], fp[0] * sun[1] - fp[1] * sun[0]};
  const float along[3] = {rp[0] + fx[0], rp[1] + fx[1], rp[2] + fx[2]};
  float turn = std::atan2(dot3(along, y0), dot3(along, x0));
  if (!std::isfinite(turn))
    turn = 0.f;
  // In steps (g_followTurnStep), held until the camera has turned three quarters of a step past one.
  if (g_followTurnStep > 0.f) {
    const float step = g_followTurnStep * 0.017453293f;
    float off = turn - f.turn;
    off -= 6.2831853f * std::round(off / 6.2831853f);
    if (!f.turnHeld || std::fabs(off) > 0.75f * step)
      f.turn = std::round(turn / step) * step;
    f.turnHeld = true;
    turn = f.turn;
  } else {
    f.turn = turn;
    f.turnHeld = true;
  }
  const float c = std::cos(turn), s = std::sin(turn);
  const float x[3] = {c * x0[0] + s * y0[0], c * x0[1] + s * y0[1], c * x0[2] + s * y0[2]};
  const float y[3] = {sun[1] * x[2] - sun[2] * x[1], sun[2] * x[0] - sun[0] * x[2], sun[0] * x[1] - sun[1] * x[0]};
  // The slices: from the near plane to HD's first split, to its second, (to the fourth's start,) to the far end.
  const float nearPlane = std::max(v[14], 1.f);
  float far = g_followFar;
  if (camera.farPlane > nearPlane)
    far = std::min(far, camera.farPlane);
  const float s0 = hd_split(0), s1 = hd_split(1);
  float bounds[kSunMapCascades + 1] = {nearPlane, s0, s1, far, far};
  uint32_t cascades = 3;
  f.split3 = 0.f;
  if (g_followSplit3 > 0.f) {
    const float s2 = std::max(g_followSplit3, 1.5f * s1);
    if (s2 < far * 0.9f) {
      bounds[3] = s2;
      cascades = 4;
      f.split3 = s2;
    }
  }
  for (uint32_t k = 1; k <= cascades; ++k)
    bounds[k] = std::max(bounds[k], bounds[k - 1] + 1.f);
  f.fadeEnd = bounds[cascades];
  f.fadeStart = f.fadeEnd * (12000.f / 14000.f);
  const float tx = v[12] * v[13], ty = v[12];
  const float size = static_cast<float>(g_sunMapSize);
  // DOL_AURORA_SUN_SHADOW_CASCADES_TRACE=1: a line whenever a cascade's size steps, and every 300th map.
  static const bool trace = std::getenv("DOL_AURORA_SUN_SHADOW_CASCADES_TRACE") != nullptr;
  static uint32_t traces = 0;
  const bool every = trace && traces++ % 300 == 0;
  for (uint32_t k = 0; k < cascades; ++k) {
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int corner = 0; corner < 8; ++corner) {
      const float d = bounds[k + ((corner >> 2) & 1)];
      const float p[3] = {((corner & 1) ? 1.f : -1.f) * tx * d, ((corner & 2) ? 1.f : -1.f) * ty * d, -d};
      float w[3];
      for (int row = 0; row < 3; ++row)
        w[row] = v[row * 4] * p[0] + v[row * 4 + 1] * p[1] + v[row * 4 + 2] * p[2] + v[row * 4 + 3];
      const float m[3] = {dot3(x, w), dot3(y, w), dot3(sun, w)};
      for (int i = 0; i < 3; ++i) {
        lo[i] = std::min(lo[i], m[i]);
        hi[i] = std::max(hi[i], m[i]);
      }
    }
    // Room for HD's lookup moved out along the normal (20 units) and the centre's snap and the filter (2 texels).
    const float pad = 1.f / (1.f - 4.f / size);
    const float before[2] = {f.held[k][0], f.held[k][1]};
    const float rx = held_size((0.5f * (hi[0] - lo[0]) + 24.f) * pad, f.held[k][0]);
    const float ry = held_size((0.5f * (hi[1] - lo[1]) + 24.f) * pad, f.held[k][1]);
    if (trace && (before[0] != rx || before[1] != ry))
      std::fprintf(stderr, "[cascades] %u resized %.0f x %.0f -> %.0f x %.0f (slice %.0f-%.0f)\n", k, 2.f * before[0],
                   2.f * before[1], 2.f * rx, 2.f * ry, bounds[k], bounds[k + 1]);
    // The centre moved by whole texels in this frame's turned frame.
    const float texelX = 2.f * rx / size, texelY = 2.f * ry / size;
    const float cx = std::round(0.5f * (lo[0] + hi[0]) / texelX) * texelX;
    const float cy = std::round(0.5f * (lo[1] + hi[1]) / texelY) * texelY;
    const float cz = 0.5f * (lo[2] + hi[2]);
    const float range = std::max(rx, ry);
    const float depth = std::max(cascade_depth(range), 0.5f * (hi[2] - lo[2]) + 0.5f * g_sunMapDepth);
    f.range[k] = range;
    f.depth[k] = depth;
    const float ix = 1.f / rx, iy = 1.f / ry, d = 1.f / (2.f * depth);
    map.world[k] = {x[0] * ix,   x[1] * ix,   x[2] * ix,   -cx * ix,  //
                    y[0] * iy,   y[1] * iy,   y[2] * iy,   -cy * iy,  //
                    -sun[0] * d, -sun[1] * d, -sun[2] * d, 0.5f + cz * d};
    if (every)
      std::fprintf(stderr, "[cascades] %u slice %.0f-%.0f box %.0f x %.0f (texel %.2f x %.2f) depth %.0f turn %.1f\n", k,
                   bounds[k], bounds[k + 1], 2.f * rx, 2.f * ry, texelX, texelY, depth, turn * 57.29578f);
  }
  for (uint32_t k = cascades; k < kSunMapCascades; ++k)
    f.range[k] = f.depth[k] = 0.f;
  map.cascades = cascades;
}

SunMap sun_map(const Camera& camera) noexcept {
  SunMap map;
  const float* v = camera.view.data();
  // The sun, or once it has gone the moon: the game puts it opposite the sun
  // (dScnKy_env_light_c's mMoonPos is the eye minus the sun's offset). As in
  // HD, the sun's shadows last while its disk is drawn, gone 8 degrees under
  // the horizon, and are cast from the sun held at 24 degrees, thinning out
  // from there; the moon is held at 2 degrees. HD's moon isn't the same both
  // ways: its shadows come in once it is 25 degrees up, full at 30, but at
  // dawn last until it is down to 8. The sun is in the
  // west (-x) at dusk, so the moon is in the west at dawn.
  const bool moon = v[18] < g_sunMapFadeLow;
  float sun[3] = {moon ? -v[17] : v[17], moon ? -v[18] : v[18], moon ? -v[19] : v[19]};
  const bool dawn = moon && sun[0] < 0.f;
  const float thin = std::clamp((sun[1] - g_sunMapFadeLow) / std::max(g_sunMapHold - g_sunMapFadeLow, 1e-3f), 0.f, 1.f);
  float fade = !moon  ? smoothstep_f(g_sunMapFadeLow, g_sunMapFadeHigh, sun[1]) * (g_sunMapDim + (1.f - g_sunMapDim) * thin)
                     : dawn ? smoothstep_f(g_moonDawnLow, g_moonDawnHigh, sun[1]) * g_moonStrength
                            : smoothstep_f(g_moonFadeLow, g_moonFadeHigh, sun[1]) * g_moonStrength;
  const float strength = g_sunMapStrength.load(std::memory_order_relaxed);
  const float hold = moon ? kMoonHold : g_sunMapHold;
  if (!moon && sun[1] < g_sunMapHold) {
    // Below the hold the sun is cast from where its path crossed it, its
    // direction fixed too, as in HD (shadows stood still from 17:20). With only
    // its height held, its bearing kept turning, the map turned with it every
    // frame and the shadows in the coarser cascades crawled. The path
    // (dScnKy setSunpos): (sin a, -cos a, -0.6 cos a), so at height h,
    // -cos a = h / sqrt(1 - 0.36 h^2) before normalising.
    const float c = g_sunMapHold / std::sqrt(1.f - 0.36f * g_sunMapHold * g_sunMapHold);
    const float x = std::copysign(std::sqrt(std::max(1.f - c * c, 0.f)), sun[0]);
    const float length = std::sqrt(x * x + c * c * 1.36f);
    sun[0] = x / length;
    sun[1] = c / length;
    sun[2] = 0.6f * c / length;
  } else if (sun[1] < hold) {
    const float across = std::sqrt(sun[0] * sun[0] + sun[2] * sun[2]);
    const float scale = across > 1e-4f ? std::sqrt(1.f - hold * hold) / across : 0.f;
    sun[0] *= scale;
    sun[1] = hold;
    sun[2] *= scale;
  }
  // Wind Waker HD's own rule from the host (aurora_set_sun_hd_light): the direction is HD's light as given
  // (held as HD holds it, the moon's side at night), the moon's strength at night, and HD's day/night switch
  // fade taking the lit side down to the shade before the light changes sides.
  SunHdLight hd;
  {
    std::lock_guard lock{g_cameraMutex};
    hd = g_sunHdLight;
  }
  if (hd.on) {
    sun[0] = v[17];
    sun[1] = v[18];
    sun[2] = v[19];
    fade = (hd.night ? g_moonStrength : 1.f) * (1.f - hd.fade);
  }
  if (!camera.valid || fade * strength <= 0.f)
    return map;
  map.light = {sun[0], sun[1], sun[2]};
  map.fade = fade * strength;
  // Across the map: perpendicular to the sun, the first level.
  const float up[3] = {0.f, std::fabs(sun[1]) > 0.99f ? 0.f : 1.f, std::fabs(sun[1]) > 0.99f ? 1.f : 0.f};
  float x[3] = {up[1] * sun[2] - up[2] * sun[1], up[2] * sun[0] - up[0] * sun[2], up[0] * sun[1] - up[1] * sun[0]};
  const float xl = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  for (float& c : x)
    c /= xl;
  float y[3] = {sun[1] * x[2] - sun[2] * x[1], sun[2] * x[0] - sun[0] * x[2], sun[0] * x[1] - sun[1] * x[0]};
  g_followState.on = g_follow && g_sunHdPrepass;
  if (g_followState.on) {
    follow_cascades(map, camera, sun, x, y);
  } else {
    // Each cascade's centre: half its range ahead of the camera, level (the view
    // looks down its -z, the inverse view's third column in the world).
    float ahead[3] = {-v[2], 0.f, -v[10]};
    const float al = std::sqrt(ahead[0] * ahead[0] + ahead[2] * ahead[2]);
    for (uint32_t cascade = 0; cascade < g_sunMapCascades; ++cascade) {
      const float range = cascade_range(cascade), depth = cascade_depth(range);
      const float centre[3] = {v[3] + (al > 1e-4f ? ahead[0] / al : 0.f) * range * 0.5f, v[7],
                               v[11] + (al > 1e-4f ? ahead[2] / al : 0.f) * range * 0.5f};
      // Moved by whole texels across the map, so its texels stay where they are on
      // the ground while the camera moves and the shadows' edges don't crawl.
      const float texel = 2.f * range / static_cast<float>(g_sunMapSize);
      const float cx = std::floor((x[0] * centre[0] + x[1] * centre[1] + x[2] * centre[2]) / texel) * texel;
      const float cy = std::floor((y[0] * centre[0] + y[1] * centre[1] + y[2] * centre[2]) / texel) * texel;
      const float cz = sun[0] * centre[0] + sun[1] * centre[1] + sun[2] * centre[2];
      const float r = 1.f / range, d = 1.f / (2.f * depth);
      map.world[cascade] = {x[0] * r,    x[1] * r,    x[2] * r,    -cx * r,  //
                            y[0] * r,    y[1] * r,    y[2] * r,    -cy * r,  //
                            -sun[0] * d, -sun[1] * d, -sun[2] * d, 0.5f + cz * d};
    }
    map.cascades = g_sunMapCascades;
  }
  for (uint32_t cascade = 0; cascade < map.cascades; ++cascade) {
    const auto& world = map.world[cascade];
    // The camera's view space there: world = A view + t, the inverse view.
    for (int row = 0; row < 3; ++row) {
      const float* w = &world[row * 4];
      for (int c = 0; c < 4; ++c) {
        float sum = c == 3 ? w[3] : 0.f;
        for (int k = 0; k < 3; ++k)
          sum += w[k] * v[k * 4 + c];
        map.caster[cascade][row * 4 + c] = sum;
      }
    }
  }
  // DOL_AURORA_SUN_SHADOW_MAP_TRACE: the sun and the camera, every 300th map.
  static const bool s_trace = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_TRACE") != nullptr;
  static uint32_t s_maps = 0;
  if (s_trace && s_maps++ % 300 == 0)
    std::fprintf(stderr, "[sun-map] sun=%.3f,%.3f,%.3f (elevation %.1f deg) eye=%.0f,%.0f,%.0f looking=%.3f,%.3f,%.3f\n",
                 sun[0], sun[1], sun[2], std::asin(std::clamp(sun[1], -1.f, 1.f)) * 57.29578f, v[3], v[7], v[11],
                 -v[2], -v[6], -v[10]);
  map.valid = true;
  return map;
}

const wgpu::TextureView& sun_map_view() {
  if (!g_sunMapTex || g_sunMapTex.GetWidth() != g_sunMapSize) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Sun shadow map",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .dimension = wgpu::TextureDimension::e2D,
        .size = {g_sunMapSize, g_sunMapSize, kSunMapCascades},
        .format = kSunMapFormat,
        .mipLevelCount = 1,
        .sampleCount = 1,
    };
    g_sunMapTex = g_device.CreateTexture(&descriptor);
    const wgpu::TextureViewDescriptor all{.dimension = wgpu::TextureViewDimension::e2DArray,
                                          .arrayLayerCount = kSunMapCascades};
    g_sunMapView = g_sunMapTex.CreateView(&all);
    for (uint32_t cascade = 0; cascade < kSunMapCascades; ++cascade) {
      const wgpu::TextureViewDescriptor layer{.dimension = wgpu::TextureViewDimension::e2D,
                                              .baseArrayLayer = cascade,
                                              .arrayLayerCount = 1};
      g_sunMapLayers[cascade] = g_sunMapTex.CreateView(&layer);
    }
  }
  return g_sunMapView;
}

const wgpu::TextureView& sun_map_layer(uint32_t cascade) {
  sun_map_view();
  return g_sunMapLayers[cascade];
}

bool torch_light_enabled() noexcept { return g_torch.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_torch_light(bool enabled) noexcept { g_torch.store(enabled, std::memory_order_relaxed); }

void set_point_lights(const float* lights, uint32_t count) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_lightCount = std::min(count, kMaxLights);
  if (g_lightCount > 0 && lights != nullptr)
    std::memcpy(g_lights.data(), lights, g_lightCount * 8 * sizeof(float));
}

bool edge_light_enabled() noexcept { return g_edge.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_edge_light(bool enabled) noexcept { g_edge.store(enabled, std::memory_order_relaxed); }

void set_exposure(float gain) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_exposureGain = gain > 0.f ? gain : 0.f;
}

bool exposure_active() noexcept {
  std::lock_guard lock{g_cameraMutex};
  return g_exposureGain > 1.f;
}

bool sea_mask_wanted() noexcept {
  return exposure_active() || ((g_sunFacing || g_sunHdPrepass) && sun_shadow_map_enabled()) || bloom_mask_wanted() ||
         rim_hd_active() || actor_light_wanted();
}

bool bloom_mask_wanted() noexcept { return g_bloomHd && g_bloomHdMask && bloom_enabled(); }

wgpu::TextureView sea_mask_target(const wgpu::Extent3D& sceneSize) {
  if (!g_seaMask || g_seaMaskSize.width != sceneSize.width || g_seaMaskSize.height != sceneSize.height) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Sea mask",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .size = {sceneSize.width, sceneSize.height, 1},
        .format = kSeaMaskFormat,
    };
    g_seaMask = g_device.CreateTexture(&descriptor);
    g_seaMaskView = g_seaMask.CreateView();
    const wgpu::TextureDescriptor depthDescriptor{
        .label = "Sea mask depth",
        // Read by the characters' shadow pre-pass (draw_actor_shadow()).
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .size = {sceneSize.width, sceneSize.height, 1},
        .format = kSeaMaskDepthFormat,
    };
    g_seaMaskDepth = g_device.CreateTexture(&depthDescriptor);
    g_seaMaskDepthView = g_seaMaskDepth.CreateView();
    g_seaMaskSize = sceneSize;
  }
  return g_seaMaskView;
}

wgpu::TextureView sea_mask_depth(const wgpu::Extent3D& sceneSize) {
  sea_mask_target(sceneSize);
  return g_seaMaskDepthView;
}

wgpu::TextureView effect_mask_target(const wgpu::Extent3D& sceneSize, wgpu::TextureFormat format) {
  if (!g_effectMask || g_effectMask.GetWidth() != sceneSize.width || g_effectMask.GetHeight() != sceneSize.height ||
      g_effectMask.GetFormat() != format) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Effects mask",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .size = {sceneSize.width, sceneSize.height, 1},
        .format = format,
    };
    g_effectMask = g_device.CreateTexture(&descriptor);
    g_effectMaskView = g_effectMask.CreateView();
  }
  g_effectMaskDrawn = true;
  return g_effectMaskView;
}

float effect_mask_push() noexcept { return g_effectPush; }

bool sun_normals_wanted() noexcept { return g_sunFacing && sun_shadow_map_enabled(); }

bool rim_hd_active() noexcept { return g_rimHd && webgpu::g_hasCoreFeatures; }

bool mask_normals_wanted() noexcept { return sun_normals_wanted() || rim_hd_active(); }

wgpu::TextureView sun_normal_target(const wgpu::Extent3D& sceneSize) {
  if (!g_sunNormal || g_sunNormalSize.width != sceneSize.width || g_sunNormalSize.height != sceneSize.height) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Sun shadows' scenery normals",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .size = {sceneSize.width, sceneSize.height, 1},
        .format = kSunNormalFormat,
    };
    g_sunNormal = g_device.CreateTexture(&descriptor);
    g_sunNormalView = g_sunNormal.CreateView();
    g_sunNormalSize = sceneSize;
  }
  return g_sunNormalView;
}

void set_sun_direction(float x, float y, float z) noexcept {
  const float length = std::sqrt(x * x + y * y + z * z);
  std::lock_guard lock{g_cameraMutex};
  if (length > 0.f) {
    g_camera[17] = x / length;
    g_camera[18] = y / length;
    g_camera[19] = z / length;
  } else {
    g_camera[17] = g_camera[18] = g_camera[19] = 0.f;
  }
}

void set_world_view(const float invView[12], float tanHalfFovy, float aspect, float nearPlane, float seaLevel,
                    float seconds) noexcept {
  std::lock_guard lock{g_cameraMutex};
  std::memcpy(g_camera.data(), invView, 12 * sizeof(float));
  g_camera[12] = tanHalfFovy;
  g_camera[13] = aspect;
  g_camera[14] = nearPlane;
  g_camera[15] = seaLevel;
  g_camera[16] = seconds;
  g_cameraValid = nearPlane > 0.f && tanHalfFovy > 0.f && aspect > 0.f;
}

void set_world_view_far(float farPlane) noexcept {
  std::lock_guard lock{g_cameraMutex};
  g_cameraFar = farPlane > 0.f ? farPlane : 0.f;
}

Camera current_camera() noexcept {
  Camera camera;
  std::lock_guard lock{g_cameraMutex};
  camera.view = g_camera;
  camera.farPlane = g_cameraFar;
  camera.exposure = g_exposureGain;
  camera.valid = g_cameraValid;
  camera.lights = g_lights;
  camera.lightCount = g_lightCount;
  return camera;
}

Camera moved_camera(const Camera& camera, const float relative[3][4], float seconds) noexcept {
  Camera moved = camera;
  for (int r = 0; r < 3; ++r) {
    const float* row = &camera.view[r * 4];
    for (int c = 0; c < 4; ++c)
      moved.view[r * 4 + c] =
          row[0] * relative[0][c] + row[1] * relative[1][c] + row[2] * relative[2][c] + (c == 3 ? row[3] : 0.f);
  }
  moved.view[16] = seconds;
  return moved;
}

bool any_enabled() noexcept {
  return bloom_enabled() || far_blur_enabled() || ambient_occlusion_enabled() || cloud_shadows_enabled() ||
         sun_shadows_enabled() || edge_light_enabled() || torch_light_enabled();
}

// The frame buffer's colours before the effects, for the exposure's
// highlights: taken after the sun shadows, a shadow on ice was darker than the
// lit ice beside it, took more of the gain and came out washed out. False
// where the target isn't the frame buffer (the exposure then has no taper).
bool copy_exposure_scene(const wgpu::CommandEncoder& cmd, const wgpu::Extent3D& size, const wgpu::TextureView& dst) {
  const auto& frame = webgpu::g_frameBuffer;
  if (!frame.texture || dst.Get() != frame.view.Get() || frame.texture.GetWidth() != size.width ||
      frame.texture.GetHeight() != size.height)
    return false;
  if (!g_exposureScene || g_exposureScene.GetWidth() != size.width || g_exposureScene.GetHeight() != size.height ||
      g_exposureScene.GetFormat() != frame.texture.GetFormat()) {
    const wgpu::TextureDescriptor descriptor{
        .label = "Exposure scene",
        .usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding,
        .size = {size.width, size.height, 1},
        .format = frame.texture.GetFormat(),
    };
    g_exposureScene = g_device.CreateTexture(&descriptor);
    g_exposureSceneView = g_exposureScene.CreateView();
  }
  const wgpu::TexelCopyTextureInfo from{.texture = frame.texture};
  const wgpu::TexelCopyTextureInfo to{.texture = g_exposureScene};
  const wgpu::Extent3D extent{size.width, size.height, 1};
  cmd.CopyTextureToTexture(&from, &to, &extent);
  return true;
}

void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
            const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
            const wgpu::TextureView& resolve, uint32_t msaaSamples, const Camera& camera, const SunMap* sunMap,
            Phase phase, const Monotone* monotone) {
  ++g_postEncodes;
  const bool exposureScene = camera.exposure > 1.f && g_module && msaaSamples == 1 && camera.valid &&
                             size.width >= 4 && size.height >= 4 && copy_exposure_scene(cmd, size, dst);
  // The effects drawn over the scene, which the shading below leaves (common.cpp
  // encode_effect_mask()), or a black texel: no effects this frame.
  effect_none_view();
  const bool effects = g_effectMaskDrawn && g_effectMask && g_effectMask.GetWidth() == size.width &&
                       g_effectMask.GetHeight() == size.height;
  g_effectMaskDrawn = false;
  const wgpu::TextureView& effectView = effects ? g_effectMaskView : g_effectNoneView;
  // The sun shadow map's pass runs this frame: with HD's SSAO and the cloud shadows on HD's rule it makes them, in
  // place of their own passes.
  const bool sunMapPass = sun_shadows_enabled() && g_module && depthView && msaaSamples == 1 && camera.valid &&
                          sunMap != nullptr && sunMap->valid && g_sunMapView && !g_sunMapDebug;
  const bool aoInSun = sunMapPass && ao_hd_on();
  const bool cloudsInSun = sunMapPass && clouds_hd_on();
  // Ambient occlusion and the far blur need single-sample depth (the depth
  // buffer is multisampled with MSAA).
  if (ambient_occlusion_enabled() && !aoInSun && g_module && depthView && msaaSamples == 1 && size.width >= 4 &&
      size.height >= 4) {
    const webgpu::gpu_prof::Zone zone{cmd, "Ambient occlusion"};
    ensure_targets(size, format, msaaSamples);
    draw_depth(cmd, "Ambient occlusion", g_aoPass, g_aoView[1], depthView, g_aoView[0], nullptr, false);
    draw_depth(cmd, "Ambient occlusion blur (horizontal)", g_aoBlurH, g_aoView[0], depthView, g_aoView[1], nullptr,
               false);
    draw_depth(cmd, "Ambient occlusion blur (vertical)", g_aoBlurV, g_aoView[1], depthView, g_aoView[0], nullptr,
               false);
    // With DOL_AURORA_AO_DEBUG=1 the occlusion replaces the scene; with 2 it tints it.
    draw_depth(cmd, "Ambient occlusion apply", g_aoUpscale, g_aoView[0], depthView, dst, resolve, true, effectView);
  }
  if (sun_shadows_enabled() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Sun shadows"};
      ensure_targets(size, format, msaaSamples);
      // The map's shadows are filtered in the map, so they are multiplied in at
      // the scene's size; at a quarter of it, blurred as the screen's version
      // is, Link's shadow was a blob.
      if (sunMap != nullptr && sunMap->valid && g_sunMapView && !g_sunMapDebug) {
        draw_sun_map(cmd, camera, *sunMap, depthView, effectView, dst, true, size, resolve);
      } else {
        if (sunMap != nullptr && sunMap->valid && g_sunMapView)
          draw_sun_map(cmd, camera, *sunMap, depthView, effectView, g_aoView[0], false, size);
        else
          draw_with_view(cmd, "Sun shadows", g_sunPass, camera.view, depthView, g_aoView[0], nullptr, &effectView);
        if (!g_sunMapDebug) {
          draw(cmd, "Sun shadows blur (horizontal)", g_blurH, g_aoView[0], g_aoView[1], false);
          draw(cmd, "Sun shadows blur (vertical)", g_blurV, g_aoView[1], g_aoView[0], false);
        }
        draw(cmd, "Sun shadows apply", g_aoApply, g_aoView[0], dst, true, resolve);
      }
    }
  }
  // Wind Waker HD's rim light on characters, in place of the edge light, wherever the characters are in the sea
  // mask (with their normals beside it).
  if (rim_hd_active() && g_module && depthView && msaaSamples == 1 && camera.valid && g_seaMaskView &&
      g_sunNormalView && g_seaMaskSize.width == size.width && g_seaMaskSize.height == size.height &&
      g_sunNormalSize.width == size.width && g_sunNormalSize.height == size.height) {
    const webgpu::gpu_prof::Zone zone{cmd, "Rim light (HD)"};
    ensure_targets(size, format, msaaSamples);
    if (!g_rimPass || g_rimSamples != msaaSamples) {
      g_rimPass = make_rim_pipeline(format, msaaSamples);
      g_rimSamples = msaaSamples;
    }
    std::array<float, 36> uniform{};
    std::memcpy(uniform.data(), camera.view.data(), sizeof(camera.view));
    {
      std::lock_guard lock{g_cameraMutex};
      for (int i = 0; i < 3; ++i) {
        uniform[20 + i] = g_sunPalette.sky[i];
        uniform[24 + i] = g_sunPalette.ground[i];
        uniform[28 + i] = g_sunPalette.lit[i];
      }
      uniform[23] = g_sunPalette.on ? 1.f : 0.f;
    }
    // The characters drawn with Wind Waker HD's light (the mask's alpha) have the rim in their draws: left out.
    uniform[27] = actor_light_applied() ? 1.f : 0.f;
    draw_with_uniform(cmd, "Rim light (HD)", g_rimPass, g_cloudLayout, uniform.data(), sizeof(uniform), depthView,
                      dst, resolve, &g_seaMaskView, &g_sunNormalView);
  } else if (edge_light_enabled() && !rim_hd_active() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Edge light"};
      ensure_targets(size, format, msaaSamples);
      draw_with_view(cmd, "Edge light", g_edgePass, camera.view, depthView, dst, resolve);
    }
  }
  // The torch glow, while Wind Waker HD's light buffer isn't drawn: with it, the characters and the lit scenery take
  // the lights in their own draws (gxcore_draw.cpp begin_actor_light_pass()).
  if (torch_light_enabled() && !(point_lights_hd_active() && actor_light_applied()) && g_module && depthView &&
      msaaSamples == 1) {
    if (camera.valid && camera.lightCount > 0) {
      std::array<float, kTorchUniformSize / sizeof(float)> uniform{};
      std::memcpy(uniform.data(), camera.view.data(), sizeof(camera.view));
      uniform[20] = static_cast<float>(camera.lightCount);
      std::memcpy(uniform.data() + 24, camera.lights.data(), sizeof(camera.lights));
      const webgpu::gpu_prof::Zone zone{cmd, "Torch light"};
      ensure_targets(size, format, msaaSamples);
      draw_with_uniform(cmd, "Torch light", g_torchPass, g_torchLayout, uniform.data(), sizeof(uniform), depthView,
                        dst, resolve);
    }
  }
  if (cloud_shadows_enabled() && !cloudsInSun && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Cloud shadows"};
      ensure_targets(size, format, msaaSamples);
      draw_with_view(cmd, "Cloud shadows", g_cloudPass, camera.view, depthView, dst, resolve, &effectView);
    }
  }
  // HD draws no far blur in a monotone frame (its DOF is the monotone's).
  if (phase == Phase::Full && far_blur_enabled() && g_module && depthView && msaaSamples == 1 && size.width >= 4 &&
      size.height >= 4) {
    const webgpu::gpu_prof::Zone zone{cmd, "Far blur"};
    ensure_targets(size, format, msaaSamples);
    draw(cmd, "Far blur downsample", g_down, scene, g_farView[0], false);
    for (int i = 0; i < 2; ++i) {
      draw(cmd, "Far blur (horizontal)", g_blurH, g_farView[0], g_farView[1], false);
      draw(cmd, "Far blur (vertical)", g_blurV, g_farView[1], g_farView[0], false);
    }
    const std::array<float, 4> far{g_farLevel.load(std::memory_order_relaxed), 0.f, 0.f, 0.f};
    draw_with_uniform(cmd, "Far blur composite", g_farComposite, g_farLayout, far.data(), sizeof far, depthView, dst,
                      resolve, &g_farView[0]);
  }
  if (camera.exposure > 1.f && g_module && size.width >= 4 && size.height >= 4) {
    const webgpu::gpu_prof::Zone zone{cmd, "Exposure"};
    ensure_targets(size, format, msaaSamples);
    // The game's brightness rides in the seconds, which the pass doesn't use.
    std::array<float, 20> view = camera.view;
    view[16] = camera.exposure;
    // And, in place of the sun's x (unused here), whether the characters took Wind Waker HD's light (fs_exposure).
    view[17] = actor_light_applied() ? 1.f : 0.f;
    if (g_exposureDepth && depthView && msaaSamples == 1 && camera.valid && g_seaMaskView)
    {
      draw_with_view(cmd, "Exposure", g_exposureDepth, view, depthView, dst, resolve, &g_seaMaskView,
                     exposureScene ? &g_exposureSceneView : &g_seaMaskView); // the mask: no taper where it is 0
    }
  }
  if (phase == Phase::Scene && monotone != nullptr && monotone->on && g_module && depthView && msaaSamples == 1 &&
      camera.valid && size.width >= 4 && size.height >= 4)
    draw_monotone(cmd, scene, depthView, size, format, dst, camera, *monotone);
  if (phase == Phase::Full && bloom_enabled())
    encode_bloom(cmd, scene, size, format, dst, resolve, msaaSamples);
}

void encode_bloom(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::Extent3D& size,
                  wgpu::TextureFormat format, const wgpu::TextureView& dst, const wgpu::TextureView& resolve,
                  uint32_t msaaSamples) {
  if (!g_module || !scene || !dst || size.width < 4 || size.height < 4) {
    return;
  }
  if (g_bloomHd && size.width >= 32 && size.height >= 32) {
    encode_bloom_hd(cmd, scene, size, format, dst, resolve, msaaSamples);
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

void aurora_set_far_blur_level(float level) { aurora::gfx::post_fx::set_far_blur_level(level); }

void aurora_set_ambient_occlusion(bool enabled) { aurora::gfx::post_fx::set_ambient_occlusion(enabled); }

void aurora_set_cloud_shadows(bool enabled) { aurora::gfx::post_fx::set_cloud_shadows(enabled); }

void aurora_set_world_view(const float inv_view[12], float tan_half_fovy, float aspect, float near_plane,
                           float sea_level, float seconds) {
  aurora::gfx::post_fx::set_world_view(inv_view, tan_half_fovy, aspect, near_plane, sea_level, seconds);
}

void aurora_set_world_view_far(float far_plane) { aurora::gfx::post_fx::set_world_view_far(far_plane); }

void aurora_set_sun_shadows(bool enabled) { aurora::gfx::post_fx::set_sun_shadows(enabled); }

void aurora_set_sun_direction(float x, float y, float z) { aurora::gfx::post_fx::set_sun_direction(x, y, z); }

void aurora_set_exposure(float gain) { aurora::gfx::post_fx::set_exposure(gain); }

void aurora_set_sun_shadow_map(bool enabled) { aurora::gfx::post_fx::set_sun_shadow_map(enabled); }

void aurora_set_sun_shadow_quality(int quality) { aurora::gfx::post_fx::set_sun_shadow_quality(quality); }

void aurora_set_sun_shadow_strength(float strength) { aurora::gfx::post_fx::set_sun_shadow_strength(strength); }
void aurora_set_sun_cascade_splits(float middle, float far) {
  aurora::gfx::post_fx::set_sun_cascade_splits(middle, far);
}
void aurora_set_sun_palette(int on, const float sky[3], const float ground[3], const float lit[3]) {
  aurora::gfx::post_fx::set_sun_palette(on != 0, sky, ground, lit);
}

void aurora_set_actor_light_hd(int on, const float c0[3], const float shadow[3], const float k[3]) {
  aurora::gfx::post_fx::set_actor_light_hd(on != 0, c0, shadow, k);
}

void aurora_set_actor_light_hd_own(int on, const float c0[3], const float k0[3]) {
  aurora::gfx::post_fx::set_actor_light_hd_own(on != 0, c0, k0);
}

void aurora_set_hd_cloud_texture(const char* path) {
  aurora::gfx::post_fx::set_hd_cloud_texture(path);
}

void aurora_set_fog_hd(int on) { aurora::gfx::post_fx::set_fog_hd(on != 0); }

void aurora_set_fog_hd_starts(const float* start_end, int count) {
  aurora::gfx::post_fx::set_fog_hd_starts(start_end, count);
}

void aurora_set_sun_hd_light(int on, int night, float fade) {
  aurora::gfx::post_fx::set_sun_hd_light(on != 0, night != 0, fade);
}

void aurora_set_cloud_hd_floor(float floor) { aurora::gfx::post_fx::set_cloud_hd_floor(floor); }
void aurora_set_sun_caster_cull(int on) { aurora::gfx::post_fx::set_sun_caster_cull(on != 0); }
void aurora_set_monotone_hd(int on, float level, float start, float end, const float mul[3], float saturate,
                            uint32_t frame_copy) {
  aurora::gfx::post_fx::set_monotone(on != 0, level, start, end, mul, saturate, frame_copy);
}
int aurora_monotone_hd_supported(void) { return aurora::gfx::post_fx::monotone_hd_supported() ? 1 : 0; }

void aurora_set_edge_light(bool enabled) { aurora::gfx::post_fx::set_edge_light(enabled); }

void aurora_set_torch_light(bool enabled) { aurora::gfx::post_fx::set_torch_light(enabled); }

void aurora_set_point_lights(const float* lights, uint32_t count) {
  aurora::gfx::post_fx::set_point_lights(lights, count);
}
