// SPDX-License-Identifier: GPL-3.0-or-later
#include <gtest/gtest.h>
#include "gfx/smooth_normal_cache.hpp"

#include <cstring>
#include <vector>

namespace {

namespace gxc = gxruntime::gxcore;
using aurora::gfx::gxcore::SmoothNormalCache;

constexpr size_t F = gxc::kVertexFloats;
constexpr size_t kColor0 = gxc::kVertexColor0Offset / sizeof(float);
constexpr size_t kNormal = gxc::kVertexNormalOffset / sizeof(float);
constexpr size_t kUv0 = gxc::kVertexUvOffset / sizeof(float);

// A draw of `count` vertices, every float its own value.
std::vector<float> make_vertices(size_t count, float seed = 0.f) {
  std::vector<float> v(count * F);
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = seed + 0.25f * static_cast<float>(i);
  return v;
}

// smooth_normals_uncached()'s stand-in: only colour 0 and the normal change, from the draw's positions and the input.
struct Maker {
  int calls = 0;
  std::vector<float> out;
  const std::vector<float>& operator()(const std::vector<float>& in, int input) {
    ++calls;
    out = in;
    for (size_t v = 0; v < in.size() / F; ++v) {
      float* vert = &out[v * F];
      for (int k = 0; k < 3; ++k) {
        vert[kNormal + k] = vert[k] * 0.5f + static_cast<float>(input);
        vert[kColor0 + k] = vert[k] * 0.125f;
      }
    }
    return out;
  }
};

const std::vector<float>& get(SmoothNormalCache<int>& cache, Maker& maker, const std::vector<float>& vertices,
                              const std::vector<uint16_t>& indices, int input, uint64_t frame) {
  return cache.get(vertices, indices, input, frame,
                   [&]() -> const std::vector<float>& { return maker(vertices, input); });
}

TEST(SmoothNormalCache, HitIsTheUncachedBytes) {
  SmoothNormalCache<int> cache;
  Maker maker;
  const auto vertices = make_vertices(40);
  const std::vector<uint16_t> indices{0, 1, 2, 2, 3, 0, 4, 5, 6};
  const std::vector<float> first = get(cache, maker, vertices, indices, 1, 10);
  const std::vector<float> uncached = Maker{}(vertices, 1);
  ASSERT_EQ(first.size(), uncached.size());
  EXPECT_EQ(std::memcmp(first.data(), uncached.data(), first.size() * sizeof(float)), 0);
  for (uint64_t frame = 11; frame < 400; ++frame) {
    const std::vector<float>& again = get(cache, maker, vertices, indices, 1, frame);
    ASSERT_EQ(std::memcmp(again.data(), uncached.data(), again.size() * sizeof(float)), 0) << frame;
  }
  EXPECT_EQ(maker.calls, 1);
  EXPECT_EQ(cache.size(), 1u);
  const auto [draws, hits] = cache.take_counts();
  EXPECT_EQ(draws, 390u);
  EXPECT_EQ(hits, 389u);
}

TEST(SmoothNormalCache, AnyChangedFloatMisses) {
  const auto base = make_vertices(40);
  const std::vector<uint16_t> indices{0, 1, 2, 3, 4, 5};
  // A position, colour 0 (made over by make(), but its own value is kept), colour 1, a texture coordinate, the normal
  // and the last float, in the middle vertex and in the last.
  for (const size_t vertex : {size_t{20}, size_t{39}}) {
    for (const size_t f : {size_t{1}, kColor0 + 1, kColor0 + 3, kColor0 + 5, kUv0 + 1, kNormal + 2, F - 1}) {
      SmoothNormalCache<int> cache;
      Maker maker;
      get(cache, maker, base, indices, 1, 5);
      auto changed = base;
      changed[vertex * F + f] += 1.f;
      const std::vector<float>& out = get(cache, maker, changed, indices, 1, 6);
      EXPECT_EQ(maker.calls, 2) << "vertex " << vertex << " float " << f;
      const std::vector<float> uncached = Maker{}(changed, 1);
      EXPECT_EQ(std::memcmp(out.data(), uncached.data(), out.size() * sizeof(float)), 0);
      // And the first draw again: its own vertices, whether its entry was the changed one's (the same key) or not.
      const std::vector<float>& back = get(cache, maker, base, indices, 1, 7);
      const std::vector<float> baseUncached = Maker{}(base, 1);
      EXPECT_EQ(std::memcmp(back.data(), baseUncached.data(), back.size() * sizeof(float)), 0);
    }
  }
}

TEST(SmoothNormalCache, ChangedIndicesOrInputsMiss) {
  SmoothNormalCache<int> cache;
  Maker maker;
  const auto vertices = make_vertices(16);
  get(cache, maker, vertices, {0, 1, 2, 3, 4, 5}, 1, 1);
  get(cache, maker, vertices, {0, 1, 2, 3, 5, 4}, 1, 2);
  EXPECT_EQ(maker.calls, 2);
  const std::vector<float>& out = get(cache, maker, vertices, {0, 1, 2, 3, 5, 4}, 2, 3);
  EXPECT_EQ(maker.calls, 3);
  const std::vector<float> uncached = Maker{}(vertices, 2);
  EXPECT_EQ(std::memcmp(out.data(), uncached.data(), out.size() * sizeof(float)), 0);
  get(cache, maker, vertices, {0, 1, 2, 3, 5, 4}, 2, 4);
  EXPECT_EQ(maker.calls, 3);
}

TEST(SmoothNormalCache, UnusedEntriesGoAfter120Frames) {
  SmoothNormalCache<int> cache;
  Maker maker;
  const auto a = make_vertices(8, 1.f), b = make_vertices(8, 2.f), c = make_vertices(8, 3.f);
  const std::vector<uint16_t> indices{0, 1, 2};
  get(cache, maker, a, indices, 1, 0);
  get(cache, maker, b, indices, 1, 0);
  // b is drawn every frame, a isn't: at frame 120 (a sweep) a is 120 frames unused, so it stays.
  for (uint64_t frame = 1; frame <= 120; ++frame)
    get(cache, maker, b, indices, 1, frame);
  EXPECT_EQ(cache.size(), 2u);
  get(cache, maker, a, indices, 1, 120);
  EXPECT_EQ(maker.calls, 2);
  // Unused from 120 on: the sweeps at 180 and 240 keep it (60, 120 frames), the one at 300 drops it (180).
  for (uint64_t frame = 121; frame <= 300; ++frame)
    get(cache, maker, b, indices, 1, frame);
  EXPECT_EQ(cache.size(), 1u);
  get(cache, maker, a, indices, 1, 301);
  EXPECT_EQ(maker.calls, 3);
  // A frame count that goes back (a new game) drops them all at the next draw.
  get(cache, maker, c, indices, 1, 5);
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_EQ(maker.calls, 4);
}

TEST(SmoothNormalCache, FullCacheMakesNewDrawsUncached) {
  using Cache = SmoothNormalCache<int>;
  Cache cache;
  Maker maker;
  const std::vector<uint16_t> indices{0, 1, 2};
  for (size_t i = 0; i < Cache::kMaxEntries; ++i)
    get(cache, maker, make_vertices(4, static_cast<float>(i)), indices, 1, 1);
  EXPECT_EQ(cache.size(), Cache::kMaxEntries);
  const auto extra = make_vertices(4, -7.f);
  const std::vector<float>& out = get(cache, maker, extra, indices, 1, 1);
  EXPECT_EQ(cache.size(), Cache::kMaxEntries);
  const std::vector<float> uncached = Maker{}(extra, 1);
  EXPECT_EQ(std::memcmp(out.data(), uncached.data(), out.size() * sizeof(float)), 0);
  const int calls = maker.calls;
  get(cache, maker, extra, indices, 1, 1);
  EXPECT_EQ(maker.calls, calls + 1);
  get(cache, maker, make_vertices(4, 0.f), indices, 1, 1);
  EXPECT_EQ(maker.calls, calls + 1);
}

} // namespace
