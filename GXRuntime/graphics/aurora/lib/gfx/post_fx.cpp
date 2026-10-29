#include "post_fx.hpp"

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

namespace aurora::gfx::post_fx {
using webgpu::g_device;

namespace {
std::atomic_bool g_bloom{false};
std::atomic_bool g_farBlur{false};
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
float g_sunReach = 1500.f;    // how far toward the sun a shadow is looked for, in world units
std::atomic_bool g_sunMapOn{false};
uint32_t g_sunMapSize = 4096;    // the shadow map's texels a side (4096 high, 2048 medium, 1024 low)
float g_sunMapRange = 1250.f;    // world units from the first cascade's centre to its edge (a texel about Link's leg)
float g_sunMapSpread = 4.f;      // each next cascade this many times wider
uint32_t g_sunMapCascades = 4;   // DOL_AURORA_SUN_SHADOW_MAP_CASCADES, 1 to kSunMapCascades
float g_sunMapDepth = 20000.f;   // world units toward and away from the sun it holds, from its centre (at least twice a cascade's range)
float g_sunMapBias = 12.f;       // world units a surface is moved toward the sun before the lookup (first cascade; scaled with the texels)
float g_sunMapContact = 300.f;   // the screen's steps' reach, for what the map leaves out
int g_sunMapDebug = 0;           // DOL_AURORA_SUN_SHADOW_MAP_DEBUG=1: the map's shadow alone, gray outside it; 2: the map itself, depth as contour bands
bool g_sunMapSteps = false;
float g_sunMapSeaLow = 40.f;      // DOL_AURORA_SUN_SHADOW_MAP_SEA_LOW: below the sea's height plus this, a surface counts as sea
float g_sunMapSeaBias = 0.f;      // DOL_AURORA_SUN_SHADOW_MAP_SEA_BIAS: the share of the slope and normal bias the sea keeps (1 before it left the map)
float g_sunMapSeaRise = 0.f;      // DOL_AURORA_SUN_SHADOW_MAP_SEA_RISE: what stands no higher than this above the sea doesn't shade it (15 before the sea left the map)
bool g_sunMapSmooth = true;
bool g_sunMapGapFill = true;      // DOL_AURORA_SUN_SHADOW_MAP_GAP_FILL=0: no steps over the bias's reach
uint32_t g_sunMapTaps = 12;       // DOL_AURORA_SUN_SHADOW_MAP_TAPS: lookups in each shadow's filter disc (6 to 12)
// The shadow quality set while running (set_sun_shadow_quality), 0 low to 2
// high, until the render thread applies it; -1 when nothing is waiting.
std::atomic_int g_sunQualityPending{-1};
float g_sunMapFarStart = 1500.f;  // DOL_AURORA_SUN_SHADOW_MAP_FAR_START/_END: world units from the camera over which the
float g_sunMapFarEnd = 5000.f;    // bias goes from its near values to its far ones
float g_sunMapNormalNear = 2.f;    // DOL_AURORA_SUN_SHADOW_MAP_NORMAL_NEAR: the normal offset near the camera, in texels
                                   // (with 0.5 and 1 below, a wall the sun only grazed was striped light and dark)
float g_sunMapSlopeNear = 10.f;   // DOL_AURORA_SUN_SHADOW_MAP_SLOPE_NEAR: the same near the camera
float g_sunMapFadeLow = -0.139f;  // DOL_AURORA_SUN_SHADOW_MAP_FADE_LOW/_HIGH: the sun's height (sin of its elevation)
float g_sunMapFadeHigh = 0.017f;  // where its shadows are gone, and at full strength (-8 and +1 degrees). The game
                                  // draws its large sun disk down to -7.5 (06:30-19:30), and HD casts while it shows
float g_sunMapHold = 0.035f;      // DOL_AURORA_SUN_SHADOW_MAP_HOLD: below this height (2 degrees) the shadows are cast
                                  // from the sun held there, as one under the horizon lights nothing
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
float g_sunMapSlope = 10.f;       // DOL_AURORA_SUN_SHADOW_MAP_SLOPE: texels of the surface's own depth slope added to the bias
float g_sunMapNormalOffset = 2.f; // DOL_AURORA_SUN_SHADOW_MAP_NORMAL_OFFSET: a lookup moved out along the surface's normal, in its cascade's texels      // DOL_AURORA_SUN_SHADOW_MAP_STEPS=1: the screen's steps too, for what the map leaves out
// A cascade's range (world units from its centre to its edge), and the depth
// toward and away from the sun it holds: enough for what stands in it.
float cascade_range(uint32_t cascade) noexcept { return g_sunMapRange * std::pow(g_sunMapSpread, static_cast<float>(cascade)); }
float cascade_depth(float range) noexcept { return std::max(g_sunMapDepth, 2.f * range); }
// The sun shadow pass's uniform (kSunMapShader's View).
constexpr size_t kSunMapUniformFloats = 76;
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
bool g_farDebug = false;
float g_threshold = 0.72f;
float g_strength = 0.35f;

wgpu::ShaderModule g_module;
wgpu::BindGroupLayout g_layout;
wgpu::PipelineLayout g_pipelineLayout;
wgpu::Sampler g_sampler;
wgpu::TextureFormat g_format = wgpu::TextureFormat::Undefined;
wgpu::RenderPipeline g_bright, g_blurH, g_blurV;
wgpu::RenderPipeline g_composite;
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

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var blurred: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;

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
    var f = select(smoothstep(far_start, far_end, 1.0 / max(d, 1e-9)) * amount, background, d <= 0.0);
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

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
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
    let shade = select(closest, sum / total, total > 1e-3);
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
constexpr const char* kCloudShader = R"""(
override strength: f32 = 0.28;
override scale: f32 = 7000.0;
override debug_view: bool = false;

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds
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

// Half an 8-bit step of noise (interleaved gradient noise), so a smooth shade
// multiplied into the 8-bit scene doesn't come out as bands on the sea.
fn dither(pos: vec2<f32>) -> f32 {
    return (fract(52.9829189 * fract(dot(pos, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
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
    let shade = smoothstep(0.42, 0.6, n) * strength * (1.0 - smoothstep(60000.0, 160000.0, distance));
    return vec4(vec3(1.0 - shade + select(0.0, dither(in.pos.xy), shade > 0.0)), 1.0);
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

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds, then the direction to the sun (world)
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

fn view_depth(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(depth));
    let p = clamp(vec2<i32>(uv * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let d = textureLoad(depth, p, 0);
    return select(view.lens.z / d, 0.0, d <= 0.0);
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
    let fade = (1.0 - smoothstep(4000.0, 9000.0, z)) * smoothstep(0.05, 0.25, sun.y);
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
override slope_texels: f32 = 10.0;
override slope_near: f32 = 10.0;
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

struct View {
    rows: array<vec4<f32>, 3>,  // the inverse view matrix (view to world), by rows
    lens: vec4<f32>,            // tan(fovy / 2), aspect, near, the sea's height
    time: vec4<f32>,            // seconds, then the direction to the sun (world)
    light: array<vec4<f32>, 12>, // the world to each cascade (post_fx::SunMap::world), three rows each
    params: vec4<f32>,          // a texel's size in the map, the cascades drawn, how strongly the shadows show, near / (far - near)
    bias: vec4<f32>,            // each cascade's depth bias, in its own depth's units
};

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var unused: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_depth_2d;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var sun_map: texture_depth_2d_array;
@group(0) @binding(5) var sun_compare: sampler_comparison;

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
    let along = abs(m.z - 0.5) * length(view.light[k * 3].xyz) / length(view.light[k * 3 + 2].xyz);
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
    let texel = 2.0 / (length(view.light[k * 3].xyz) * f32(textureDimensions(sun_map).x));
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
    let texel = 2.0 / (length(view.light[k * 3].xyz) * size.x);
    let per_unit = length(view.light[k * 3 + 2].xyz);  // the map's depth per world unit along the sun
    let reference = m.z - (view.bias[k] + texel * slope * per_unit) * grow;
    // Nothing at the sea's height or below shades the water or the sand at its
    // edge: the islands' models carry their sea floor up to the waterline, and
    // the waves dip below it, so the sea read it as something above.
    // How much the surface counts as sea: fully below sea_low over the sea's
    // height, not at all 40 units higher (no hard line for the waves to cross).
    let low = 1.0 - smoothstep(sea_low, sea_low + 40.0, world_y - view.lens.w);
    var sum = 0.0;
    let n = i32(taps);
    for (var i = 0; i < n; i++) {
        var o = vec2(0.0);
        if (i > 0) {
            let r = 2.0 * sqrt((f32(i - 1) + 0.5) / (taps - 1.0));
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

@fragment
fn fs_sun_map(in: VertexOutput) -> @location(0) vec4<f32> {
    let z = view_depth(in.uv);
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
    let k = first_cascade(world);
    if (k >= 0) {
        let turn = 6.2831853 * fract(52.9829189 * fract(dot(in.pos.xy, vec2(0.06711056, 0.00583715))));
        let rot = mat2x2<f32>(cos(turn), sin(turn), -sin(turn), cos(turn));
        let grow = grow_at(k, world);
        // The sea casts nothing, so it can't shade itself and needs neither
        // the normal offset nor the slope bias: with them, a tower standing in
        // the water at a low sun had its shadow start well away from its foot.
        let unbiased = mix(1.0, sea_bias, sea);
        let offset = mix(normal_near, normal_offset, far) * unbiased;
        let m = to_cascade(k, offset_world(k, world, push * offset * grow));
        // tan of the angle between the surface and the sun (capped), times how
        // many texels of it the bias takes: few near the camera, so shadows
        // stay at their casters' feet, more far off, where the surface's
        // direction from the depth buffer is rough.
        let ndl = dot(normal, sun);
        let cap = mix(4.0, 8.0, far);
        let tilt = select(cap, min(sqrt(max(1.0 - ndl * ndl, 0.0)) / ndl, cap), ndl > 0.0);
        let slope = tilt * mix(slope_near, slope_texels, far) * unbiased;
        lit = lit_in(k, m, rot, slope, world.y, grow);
        // Near a cascade's edge, blend into the next (or, past the last, into
        // the light), so the step in sharpness doesn't show as a line.
        let edge = reach(k, m);
        let blend = smoothstep(0.85, 1.0, edge);
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
        // 6: each cascade tinted (red, green, blue, yellow), to see where they reach.
        if (debug_view > 5.5) {
            var tint = array<vec3<f32>, 4>(vec3(1.0, 0.6, 0.6), vec3(0.6, 1.0, 0.6), vec3(0.6, 0.6, 1.0), vec3(1.0, 1.0, 0.5))[k];
            return vec4(tint * (0.4 + 0.6 * lit), 1.0);
        }
    } else if (debug_view > 0.5) {
        return vec4(vec3(0.5), 1.0);
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
    let fade = view.params.z;
    let shade = 1.0 - strength * (1.0 - lit) * fade;
    // Multiplied into the 8-bit scene: half a step of noise against bands (fs_ao_apply).
    let noise = (fract(52.9829189 * fract(dot(in.pos.xy, vec2(0.06711056, 0.00583715)))) - 0.5) * (1.5 / 255.0);
    return vec4(vec3(shade + select(0.0, noise, direct > 0.5 && shade < 0.999)), 1.0);
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
    lights: array<Light, 16>,
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
    for (var i = 0u; i < 16u; i++) {
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

wgpu::RenderPipeline make_cloud_pipeline(wgpu::TextureFormat format, uint32_t samples) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_cloudStrength},
      wgpu::ConstantEntry{.key = "scale", .value = g_cloudScale},
      wgpu::ConstantEntry{.key = "debug_view", .value = g_cloudDebug ? 1.0 : 0.0},
  };
  const wgpu::BlendState multiply{
      .color = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::Src},
      .alpha = {.operation = wgpu::BlendOperation::Add,
                .srcFactor = wgpu::BlendFactor::Zero,
                .dstFactor = wgpu::BlendFactor::One},
  };
  const std::array targets{wgpu::ColorTargetState{.format = format, .blend = &multiply, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_cloudModule,
      .entryPoint = "fs_clouds",
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = targets.size(),
      .targets = targets.data(),
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = "Cloud shadows",
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

// direct: multiplied into the scene (the EFB's alpha kept), else written to a target of its own.
wgpu::RenderPipeline make_sun_map_pipeline(wgpu::TextureFormat format, bool direct) {
  const std::array constants{
      wgpu::ConstantEntry{.key = "strength", .value = g_sunStrength},
      wgpu::ConstantEntry{.key = "contact", .value = g_sunMapContact},
      wgpu::ConstantEntry{.key = "debug_view", .value = static_cast<double>(g_sunMapDebug)},
      wgpu::ConstantEntry{.key = "screen_steps", .value = g_sunMapSteps ? 1.0 : 0.0},
      wgpu::ConstantEntry{.key = "normal_offset", .value = g_sunMapNormalOffset},
      wgpu::ConstantEntry{.key = "slope_texels", .value = g_sunMapSlope},
      wgpu::ConstantEntry{.key = "slope_near", .value = g_sunMapSlopeNear},
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
      wgpu::ColorTargetState{.format = format, .blend = direct ? &multiply : nullptr, .writeMask = wgpu::ColorWriteMask::All}};
  const wgpu::FragmentState fragment{
      .module = g_sunMapModule,
      .entryPoint = "fs_sun_map",
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
      .layout = g_depthPipelineLayout,
      .vertex = {.module = g_farModule, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .multisample = {.count = samples, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  return g_device.CreateRenderPipeline(&descriptor);
}

// A shadow quality's map size, cascades, filter lookups and gap fill (0 low,
// 1 medium, 2 high), then each of their own variables on top. Low keeps
// medium's sharpness (at 1024 every texel term grew and small shapes became
// blobs) and saves a cascade, half the lookups and the gap fill's steps.
void apply_sun_quality(int level) {
  g_sunMapSize = level >= 2 ? 4096 : 2048;
  g_sunMapCascades = level <= 0 ? 3 : kSunMapCascades;
  g_sunMapTaps = level <= 0 ? 6 : 12;
  g_sunMapGapFill = level > 0;
  g_sunMapTaps = static_cast<uint32_t>(env_float("DOL_AURORA_SUN_SHADOW_MAP_TAPS", static_cast<float>(g_sunMapTaps), 6.f, 12.f));
  g_sunMapSize = static_cast<uint32_t>(env_float("DOL_AURORA_SUN_SHADOW_MAP_SIZE", static_cast<float>(g_sunMapSize), 256.f, 8192.f));
  g_sunMapCascades = static_cast<uint32_t>(env_float("DOL_AURORA_SUN_SHADOW_MAP_CASCADES",
                                                     static_cast<float>(g_sunMapCascades), 1.f,
                                                     static_cast<float>(kSunMapCascades)));
  if (const char* gapFill = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_GAP_FILL"); gapFill != nullptr)
    g_sunMapGapFill = gapFill[0] != '0';
}

void ensure_targets(const wgpu::Extent3D& sceneSize, wgpu::TextureFormat format, uint32_t samples) {
  // A quality set while running: its map is made at its size when next used
  // (sun_map_view), and the two pipelines that take its lookups and gap fill
  // as constants are made again below.
  if (const int level = g_sunQualityPending.exchange(-1, std::memory_order_relaxed); level >= 0) {
    apply_sun_quality(level);
    g_sunMapPass = {};
    g_sunMapDirect = {};
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
    g_edgePass = {};
    g_torchPass = {};
    g_composite = {};
    g_farComposite = {};
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
void draw_depth(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                const wgpu::TextureView& source, const wgpu::TextureView& depthView, const wgpu::TextureView& target,
                const wgpu::TextureView& resolve, bool load) {
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = source},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
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
                       const wgpu::TextureView& resolve);

void draw_with_view(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                    const std::array<float, 20>& view, const wgpu::TextureView& depthView,
                    const wgpu::TextureView& target, const wgpu::TextureView& resolve) {
  draw_with_uniform(cmd, label, pipeline, g_cloudLayout, view.data(), sizeof(view), depthView, target, resolve);
}

void draw_with_uniform(const wgpu::CommandEncoder& cmd, const char* label, const wgpu::RenderPipeline& pipeline,
                       const wgpu::BindGroupLayout& layout, const void* uniform, size_t uniformSize,
                       const wgpu::TextureView& depthView, const wgpu::TextureView& target,
                       const wgpu::TextureView& resolve) {
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
      wgpu::BindGroupEntry{.binding = 1, .textureView = g_farView[0]},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .buffer = buffer, .offset = 0, .size = uniformSize},
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
void draw_sun_map(const wgpu::CommandEncoder& cmd, const Camera& camera, const SunMap& sunMap,
                  const wgpu::TextureView& depthView, const wgpu::TextureView& target, bool direct,
                  const wgpu::TextureView& resolve = nullptr) {
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
    uniform[72 + cascade] = g_sunMapBias * (range / 5000.f) / (2.f * cascade_depth(range));
  }
  const wgpu::BufferDescriptor bufferDescriptor{
      .label = "Sun shadow map view",
      .usage = wgpu::BufferUsage::Uniform,
      .size = sizeof(uniform),
      .mappedAtCreation = true,
  };
  const auto buffer = g_device.CreateBuffer(&bufferDescriptor);
  std::memcpy(buffer.GetMappedRange(0, sizeof(uniform)), uniform.data(), sizeof(uniform));
  buffer.Unmap();
  const std::array entries{
      wgpu::BindGroupEntry{.binding = 0, .sampler = g_sampler},
      wgpu::BindGroupEntry{.binding = 1, .textureView = g_farView[0]},
      wgpu::BindGroupEntry{.binding = 2, .textureView = depthView},
      wgpu::BindGroupEntry{.binding = 3, .buffer = buffer, .offset = 0, .size = sizeof(uniform)},
      wgpu::BindGroupEntry{.binding = 4, .textureView = g_sunMapView},
      wgpu::BindGroupEntry{.binding = 5, .sampler = g_sunMapCompare},
  };
  const wgpu::BindGroupDescriptor bindGroupDescriptor{
      .layout = g_sunMapLayout,
      .entryCount = entries.size(),
      .entries = entries.data(),
  };
  const auto bindGroup = g_device.CreateBindGroup(&bindGroupDescriptor);
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
} // namespace

void initialize() {
  const char* bloom = std::getenv("DOL_AURORA_BLOOM");
  g_bloom.store(bloom != nullptr && bloom[0] == '1', std::memory_order_relaxed);
  g_threshold = env_float("DOL_AURORA_BLOOM_THRESHOLD", g_threshold, 0.f, 1.f);
  g_strength = env_float("DOL_AURORA_BLOOM_STRENGTH", g_strength, 0.f, 4.f);
  const char* far = std::getenv("DOL_AURORA_FAR_BLUR");
  g_farBlur.store(far != nullptr && far[0] == '1', std::memory_order_relaxed);
  g_farStart = env_float("DOL_AURORA_FAR_BLUR_START", g_farStart, 1.f, 1e9f);
  g_farEnd = env_float("DOL_AURORA_FAR_BLUR_END", g_farEnd, g_farStart + 1.f, 1e9f);
  g_farAmount = env_float("DOL_AURORA_FAR_BLUR_AMOUNT", g_farAmount, 0.f, 1.f);
  g_farBackground = env_float("DOL_AURORA_FAR_BLUR_BACKGROUND", g_farBackground, 0.f, 1.f);
  const char* ao = std::getenv("DOL_AURORA_AO");
  g_ao.store(ao != nullptr && ao[0] == '1', std::memory_order_relaxed);
  g_aoRadius = env_float("DOL_AURORA_AO_RADIUS", g_aoRadius, 0.1f, 1e6f);
  g_aoStrength = env_float("DOL_AURORA_AO_STRENGTH", g_aoStrength, 0.f, 1.f);
  g_aoFadeStart = env_float("DOL_AURORA_AO_FADE_START", g_aoFadeStart, 1.f, 1e9f);
  g_aoFadeEnd = env_float("DOL_AURORA_AO_FADE_END", g_aoFadeEnd, g_aoFadeStart + 1.f, 1e9f);
  g_cloudStrength = env_float("DOL_AURORA_CLOUD_SHADOWS_STRENGTH", g_cloudStrength, 0.f, 1.f);
  g_cloudScale = env_float("DOL_AURORA_CLOUD_SHADOWS_SCALE", g_cloudScale, 1.f, 1e7f);
  g_sunStrength = env_float("DOL_AURORA_SUN_SHADOWS_STRENGTH", g_sunStrength, 0.f, 1.f);
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
    apply_sun_quality(quality == nullptr ? 2 : std::strcmp(quality, "low") == 0 ? 0 : std::strcmp(quality, "medium") == 0 ? 1 : 2);
  }
  g_sunMapRange = env_float("DOL_AURORA_SUN_SHADOW_MAP_RANGE", g_sunMapRange, 500.f, 100000.f);
  g_sunMapSpread = env_float("DOL_AURORA_SUN_SHADOW_MAP_SPREAD", g_sunMapSpread, 1.5f, 16.f);
  g_sunMapDepth = env_float("DOL_AURORA_SUN_SHADOW_MAP_DEPTH", g_sunMapDepth, 1000.f, 1000000.f);
  g_sunMapBias = env_float("DOL_AURORA_SUN_SHADOW_MAP_BIAS", g_sunMapBias, 0.f, 1000.f);
  g_sunMapSlope = env_float("DOL_AURORA_SUN_SHADOW_MAP_SLOPE", g_sunMapSlope, 0.f, 32.f);
  g_sunMapFadeLow = env_float("DOL_AURORA_SUN_SHADOW_MAP_FADE_LOW", g_sunMapFadeLow, -0.5f, 0.9f);
  g_sunMapFadeHigh = env_float("DOL_AURORA_SUN_SHADOW_MAP_FADE_HIGH", g_sunMapFadeHigh, g_sunMapFadeLow + 0.001f, 1.f);
  g_sunMapHold = env_float("DOL_AURORA_SUN_SHADOW_MAP_HOLD", g_sunMapHold, 0.f, 0.5f);
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
  const char* sunMapDebug = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_DEBUG");
  g_sunMapDebug = sunMapDebug != nullptr ? sunMapDebug[0] - '0' : 0;
  const char* sunMapSteps = std::getenv("DOL_AURORA_SUN_SHADOW_MAP_STEPS");
  g_sunMapSteps = sunMapSteps != nullptr && sunMapSteps[0] == '1';
  g_sunMapSeaLow = env_float("DOL_AURORA_SUN_SHADOW_MAP_SEA_LOW", g_sunMapSeaLow, 0.f, 1000.f);
  g_sunMapSeaRise = env_float("DOL_AURORA_SUN_SHADOW_MAP_SEA_RISE", g_sunMapSeaRise, 0.f, 1000.f);
  g_sunMapSeaBias = env_float("DOL_AURORA_SUN_SHADOW_MAP_SEA_BIAS", g_sunMapSeaBias, 0.f, 1.f);
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
  cloudSource.code = kCloudShader;
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
  const std::array cloudEntries{
      depthEntries[0],
      depthEntries[1],
      depthEntries[2],
      wgpu::BindGroupLayoutEntry{
          .binding = 3,
          .visibility = wgpu::ShaderStage::Fragment,
          .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = sizeof(g_camera)},
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
  sunMapSource.code = kSunMapShader;
  const wgpu::ShaderModuleDescriptor sunMapModuleDescriptor{.nextInChain = &sunMapSource,
                                                            .label = "Sun shadows (shadow map)"};
  g_sunMapModule = g_device.CreateShaderModule(&sunMapModuleDescriptor);
  std::array<wgpu::BindGroupLayoutEntry, 6> sunMapEntries{
      cloudEntries[0],
      cloudEntries[1],
      cloudEntries[2],
      cloudEntries[3],
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
}

void shutdown() {
  g_bright = g_blurH = g_blurV = g_composite = g_down = g_farComposite = g_aoPass = g_aoApply = {};
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
  g_sunMapModule = {};
  g_sunMapPipelineLayout = {};
  g_sunMapLayout = {};
  g_sunMapCompare = {};
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
  g_tex = {};
  g_view = {};
  g_farTex = {};
  g_farView = {};
  g_farSamples = 0;
  g_depthPipelineLayout = {};
  g_depthLayout = {};
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

bool ambient_occlusion_enabled() noexcept { return g_ao.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_ambient_occlusion(bool enabled) noexcept { g_ao.store(enabled, std::memory_order_relaxed); }

bool cloud_shadows_enabled() noexcept { return g_clouds.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_cloud_shadows(bool enabled) noexcept { g_clouds.store(enabled, std::memory_order_relaxed); }

bool sun_shadows_enabled() noexcept { return g_sun.load(std::memory_order_relaxed) && webgpu::g_hasCoreFeatures; }

void set_sun_shadows(bool enabled) noexcept { g_sun.store(enabled, std::memory_order_relaxed); }

bool sun_shadow_map_enabled() noexcept { return sun_shadows_enabled() && g_sunMapOn.load(std::memory_order_relaxed); }

void set_sun_shadow_map(bool enabled) noexcept { g_sunMapOn.store(enabled, std::memory_order_relaxed); }

void set_sun_shadow_quality(int level) noexcept {
  if (level >= 0 && level <= 2)
    g_sunQualityPending.store(level, std::memory_order_relaxed);
}

SunMap sun_map(const Camera& camera) noexcept {
  SunMap map;
  const float* v = camera.view.data();
  // The sun, or once it has gone the moon: the game puts it opposite the sun
  // (dScnKy_env_light_c's mMoonPos is the eye minus the sun's offset). As in
  // HD, the sun's shadows last while its disk is drawn, fading out from the
  // horizon to 8 degrees under it, cast from the sun held a little up. HD's
  // moon isn't the same both ways: its shadows come in once it is 25 degrees
  // up, full at 30, but at dawn last until it is down to 8. The sun is in the
  // west (-x) at dusk, so the moon is in the west at dawn.
  const bool moon = v[18] < g_sunMapFadeLow;
  float sun[3] = {moon ? -v[17] : v[17], moon ? -v[18] : v[18], moon ? -v[19] : v[19]};
  const bool dawn = moon && sun[0] < 0.f;
  const float fade = !moon  ? smoothstep_f(g_sunMapFadeLow, g_sunMapFadeHigh, sun[1])
                     : dawn ? smoothstep_f(g_moonDawnLow, g_moonDawnHigh, sun[1]) * g_moonStrength
                            : smoothstep_f(g_moonFadeLow, g_moonFadeHigh, sun[1]) * g_moonStrength;
  if (sun[1] < g_sunMapHold) {
    const float across = std::sqrt(sun[0] * sun[0] + sun[2] * sun[2]);
    const float scale = across > 1e-4f ? std::sqrt(1.f - g_sunMapHold * g_sunMapHold) / across : 0.f;
    sun[0] *= scale;
    sun[1] = g_sunMapHold;
    sun[2] *= scale;
  }
  if (!camera.valid || fade <= 0.f)
    return map;
  map.light = {sun[0], sun[1], sun[2]};
  map.fade = fade;
  // Across the map: perpendicular to the sun, the first level.
  const float up[3] = {0.f, std::fabs(sun[1]) > 0.99f ? 0.f : 1.f, std::fabs(sun[1]) > 0.99f ? 1.f : 0.f};
  float x[3] = {up[1] * sun[2] - up[2] * sun[1], up[2] * sun[0] - up[0] * sun[2], up[0] * sun[1] - up[1] * sun[0]};
  const float xl = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  for (float& c : x)
    c /= xl;
  const float y[3] = {sun[1] * x[2] - sun[2] * x[1], sun[2] * x[0] - sun[0] * x[2], sun[0] * x[1] - sun[1] * x[0]};
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
    auto& world = map.world[cascade];
    world = {x[0] * r,    x[1] * r,    x[2] * r,    -cx * r,  //
             y[0] * r,    y[1] * r,    y[2] * r,    -cy * r,  //
             -sun[0] * d, -sun[1] * d, -sun[2] * d, 0.5f + cz * d};
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
  map.cascades = g_sunMapCascades;
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

void encode(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::TextureView& depthView,
            const wgpu::Extent3D& size, wgpu::TextureFormat format, const wgpu::TextureView& dst,
            const wgpu::TextureView& resolve, uint32_t msaaSamples, const Camera& camera, const SunMap* sunMap) {
  // Ambient occlusion and the far blur need single-sample depth (the depth
  // buffer is multisampled with MSAA).
  if (ambient_occlusion_enabled() && g_module && depthView && msaaSamples == 1 && size.width >= 4 && size.height >= 4) {
    const webgpu::gpu_prof::Zone zone{cmd, "Ambient occlusion"};
    ensure_targets(size, format, msaaSamples);
    draw_depth(cmd, "Ambient occlusion", g_aoPass, g_aoView[1], depthView, g_aoView[0], nullptr, false);
    draw_depth(cmd, "Ambient occlusion blur (horizontal)", g_aoBlurH, g_aoView[0], depthView, g_aoView[1], nullptr,
               false);
    draw_depth(cmd, "Ambient occlusion blur (vertical)", g_aoBlurV, g_aoView[1], depthView, g_aoView[0], nullptr,
               false);
    // With DOL_AURORA_AO_DEBUG=1 the occlusion replaces the scene; with 2 it tints it.
    draw_depth(cmd, "Ambient occlusion apply", g_aoUpscale, g_aoView[0], depthView, dst, resolve, true);
  }
  if (sun_shadows_enabled() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Sun shadows"};
      ensure_targets(size, format, msaaSamples);
      // The map's shadows are filtered in the map, so they are multiplied in at
      // the scene's size; at a quarter of it, blurred as the screen's version
      // is, Link's shadow was a blob.
      if (sunMap != nullptr && sunMap->valid && g_sunMapView && !g_sunMapDebug) {
        draw_sun_map(cmd, camera, *sunMap, depthView, dst, true, resolve);
      } else {
        if (sunMap != nullptr && sunMap->valid && g_sunMapView)
          draw_sun_map(cmd, camera, *sunMap, depthView, g_aoView[0], false);
        else
          draw_with_view(cmd, "Sun shadows", g_sunPass, camera.view, depthView, g_aoView[0], nullptr);
        if (!g_sunMapDebug) {
          draw(cmd, "Sun shadows blur (horizontal)", g_blurH, g_aoView[0], g_aoView[1], false);
          draw(cmd, "Sun shadows blur (vertical)", g_blurV, g_aoView[1], g_aoView[0], false);
        }
        draw(cmd, "Sun shadows apply", g_aoApply, g_aoView[0], dst, true, resolve);
      }
    }
  }
  if (edge_light_enabled() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Edge light"};
      ensure_targets(size, format, msaaSamples);
      draw_with_view(cmd, "Edge light", g_edgePass, camera.view, depthView, dst, resolve);
    }
  }
  if (torch_light_enabled() && g_module && depthView && msaaSamples == 1) {
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
  if (cloud_shadows_enabled() && g_module && depthView && msaaSamples == 1) {
    if (camera.valid) {
      const webgpu::gpu_prof::Zone zone{cmd, "Cloud shadows"};
      ensure_targets(size, format, msaaSamples);
      draw_with_view(cmd, "Cloud shadows", g_cloudPass, camera.view, depthView, dst, resolve);
    }
  }
  if (far_blur_enabled() && g_module && depthView && msaaSamples == 1 && size.width >= 4 && size.height >= 4) {
    const webgpu::gpu_prof::Zone zone{cmd, "Far blur"};
    ensure_targets(size, format, msaaSamples);
    draw(cmd, "Far blur downsample", g_down, scene, g_farView[0], false);
    for (int i = 0; i < 2; ++i) {
      draw(cmd, "Far blur (horizontal)", g_blurH, g_farView[0], g_farView[1], false);
      draw(cmd, "Far blur (vertical)", g_blurV, g_farView[1], g_farView[0], false);
    }
    draw_depth(cmd, "Far blur composite", g_farComposite, g_farView[0], depthView, dst, resolve, true);
  }
  if (bloom_enabled())
    encode_bloom(cmd, scene, size, format, dst, resolve, msaaSamples);
}

void encode_bloom(const wgpu::CommandEncoder& cmd, const wgpu::TextureView& scene, const wgpu::Extent3D& size,
                  wgpu::TextureFormat format, const wgpu::TextureView& dst, const wgpu::TextureView& resolve,
                  uint32_t msaaSamples) {
  if (!g_module || !scene || !dst || size.width < 4 || size.height < 4) {
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

void aurora_set_ambient_occlusion(bool enabled) { aurora::gfx::post_fx::set_ambient_occlusion(enabled); }

void aurora_set_cloud_shadows(bool enabled) { aurora::gfx::post_fx::set_cloud_shadows(enabled); }

void aurora_set_world_view(const float inv_view[12], float tan_half_fovy, float aspect, float near_plane,
                           float sea_level, float seconds) {
  aurora::gfx::post_fx::set_world_view(inv_view, tan_half_fovy, aspect, near_plane, sea_level, seconds);
}

void aurora_set_world_view_far(float far_plane) { aurora::gfx::post_fx::set_world_view_far(far_plane); }

void aurora_set_sun_shadows(bool enabled) { aurora::gfx::post_fx::set_sun_shadows(enabled); }

void aurora_set_sun_direction(float x, float y, float z) { aurora::gfx::post_fx::set_sun_direction(x, y, z); }

void aurora_set_sun_shadow_map(bool enabled) { aurora::gfx::post_fx::set_sun_shadow_map(enabled); }

void aurora_set_sun_shadow_quality(int quality) { aurora::gfx::post_fx::set_sun_shadow_quality(quality); }

void aurora_set_edge_light(bool enabled) { aurora::gfx::post_fx::set_edge_light(enabled); }

void aurora_set_torch_light(bool enabled) { aurora::gfx::post_fx::set_torch_light(enabled); }

void aurora_set_point_lights(const float* lights, uint32_t count) {
  aurora::gfx::post_fx::set_point_lights(lights, count);
}
