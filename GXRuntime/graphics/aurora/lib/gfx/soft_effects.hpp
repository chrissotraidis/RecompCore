#pragma once

// Efficient soft effects (aurora_set_efficient_effects): haze, smoke, fog and
// clouds the game draws as large see-through quads after the scene. At a high
// render scale each blends over millions of pixels, several layers deep (the
// Lava Room's heat haze, 81 quads, was two thirds of its GPU time at scale 3 on
// a Steam Deck). A run of them is drawn into a buffer of half the EFB's size
// and blended over the EFB once, in the same place in the draw order.
//
// The buffer starts at (0,0,0,0) and each quad blends into it with the colour
// factors (SrcAlpha, OneMinusSrcAlpha) and the alpha factors (One,
// OneMinusSrcAlpha), so it holds the run's colour premultiplied by its coverage
// and the coverage. Blending it over the EFB with (One, OneMinusSrcAlpha) is
// then the same as blending each quad onto the EFB in turn; only the
// resolution differs.

#include <gxruntime/gxcore/shader.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aurora::gfx::soft_effects {

namespace gxc = gxruntime::gxcore;

// The fewest draws in a row worth a buffer of their own: a few sprites (a
// torch's flame) keep the EFB.
constexpr std::size_t kMinRun = 8;
// The narrowest EFB (in pixels) that takes the path: render scale 2 and up.
constexpr std::uint32_t kMinWidth = 1280;

// A soft-effect quad's pipeline: blended with source alpha over inverse source
// alpha, no depth test or depth write, colour written, triangles, one TEV
// stage and no indirect stages, and no destination alpha. Decided when the draw
// is recorded (the key is known there).
inline bool qualifies_key(const gxc::PipelineKey& key) noexcept {
  return key.blend_enable != 0 && key.blend_subtract == 0 &&
         key.src_factor == static_cast<std::uint8_t>(gxc::SrcBlendFactor::SrcAlpha) &&
         key.dst_factor == static_cast<std::uint8_t>(gxc::DstBlendFactor::InvSrcAlpha) &&
         key.depth_test == 0 && key.depth_update == 0 && key.color_update != 0 &&
         key.primitive_topology == 0 && key.shader.num_tev_stages == 1 && key.shader.num_ind_stages == 0 &&
         key.shader.use_dst_alpha == 0;
}

// And its size: one or two triangles for each draw of a batch (consecutive
// quads of one state are merged into one draw after recording). Decided when
// the frame is encoded.
inline bool qualifies_size(std::uint32_t indexCount, std::uint32_t batchSize) noexcept {
  return indexCount != 0 && batchSize != 0 && indexCount <= 12u * batchSize;
}

struct Run {
  std::size_t first = 0; // index of its first draw
  std::size_t count = 0;
};

// The runs of at least `minRun` consecutive qualifying draws, in order. A draw
// that does not qualify (or a run's change of alpha writing, which the one
// composite cannot follow) ends a run.
inline std::vector<Run> find_runs(const std::vector<bool>& soft, const std::vector<bool>& writesAlpha,
                                  std::size_t minRun = kMinRun) {
  std::vector<Run> runs;
  std::size_t i = 0;
  while (i < soft.size()) {
    if (!soft[i]) {
      ++i;
      continue;
    }
    std::size_t end = i + 1;
    while (end < soft.size() && soft[end] && writesAlpha[end] == writesAlpha[i])
      ++end;
    if (end - i >= minRun)
      runs.push_back({.first = i, .count = end - i});
    i = end;
  }
  return runs;
}

} // namespace aurora::gfx::soft_effects
