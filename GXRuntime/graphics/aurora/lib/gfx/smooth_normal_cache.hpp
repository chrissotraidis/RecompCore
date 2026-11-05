#pragma once

#include <gxruntime/gxcore/shader.hpp> // the vertex layout

#include <absl/container/flat_hash_map.h>
#include <xxhash.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace aurora::gfx::gxcore {

// The sun shadows' smooth normals (gxcore_draw.cpp with_smooth_normals()) kept from frame to frame. A draw's smooth
// normals and HD colours are a function of its vertices and indices and of `Inputs` (the stage's and the objects'
// tables, and for HD's light the draw's shade and light colours), so a still room's draws (Hyrule Castle's hall: 66
// draws and 47,000 welded, looked-up vertices a frame) make them once. A hit is the same bytes in and the same inputs,
// compared in full (the hash only picks the entry), so the vertices out are the very ones make() made for them.
// make() may change only colour 0 and the normal (smooth_normals_uncached() writes nothing else but a lava draw's
// colour 1, and lava draws don't come here): an entry keeps the rest of the draw's vertices in `out` and only those two
// of its own beside it, so the cache holds about one copy of the room's vertices, not two. Were make() to change
// anything else, the draw would never match again and would be made every frame, as without the cache.
// Entries not used for 120 game frames go (looked at every 60); past 1024 entries a new draw isn't kept.
template <typename Inputs>
class SmoothNormalCache {
public:
  static constexpr uint64_t kSweepFrames = 60;
  static constexpr uint64_t kKeepFrames = 120;
  static constexpr size_t kMaxEntries = 1024;

  template <typename Make>
  const std::vector<float>& get(const std::vector<float>& vertices, const std::vector<uint16_t>& indices,
                                Inputs inputs, uint64_t frame, Make&& make) {
    ++draws_;
    if (frame - lastSweep_ >= kSweepFrames) {
      lastSweep_ = frame;
      absl::erase_if(entries_, [&](const auto& kv) { return frame - kv.second.used > kKeepFrames; });
    }
    const uint64_t key = key_of(vertices, indices);
    auto it = entries_.find(key);
    if (it != entries_.end() && it->second.inputs == inputs && same_draw(it->second, vertices, indices)) {
      it->second.used = frame;
      ++hits_;
      return it->second.out;
    }
    if (it == entries_.end()) {
      if (entries_.size() >= kMaxEntries)
        return make();
      it = entries_.try_emplace(key).first;
    }
    Entry& entry = it->second;
    entry.out = make();
    const size_t count = vertices.size() / gxruntime::gxcore::kVertexFloats;
    entry.own.resize(count * kOwnFloats);
    for (size_t v = 0; v < count; ++v) {
      const float* in = &vertices[v * gxruntime::gxcore::kVertexFloats];
      std::memcpy(&entry.own[v * kOwnFloats], in + kColor0Float, kColor0Floats * sizeof(float));
      std::memcpy(&entry.own[v * kOwnFloats + kColor0Floats], in + kNormalFloat, kNormalFloats * sizeof(float));
    }
    entry.indices.assign(indices.begin(), indices.end());
    entry.inputs = std::move(inputs);
    entry.used = frame;
    return entry.out;
  }

  size_t size() const { return entries_.size(); }
  // Draws asked for and draws found, since the last take_counts().
  std::pair<uint64_t, uint64_t> take_counts() { return {std::exchange(draws_, 0), std::exchange(hits_, 0)}; }

private:
  static constexpr size_t kColor0Float = gxruntime::gxcore::kVertexColor0Offset / sizeof(float);
  static constexpr size_t kColor0Floats = 4;
  static constexpr size_t kNormalFloat = gxruntime::gxcore::kVertexNormalOffset / sizeof(float);
  static constexpr size_t kNormalFloats = 3;
  static constexpr size_t kOwnFloats = kColor0Floats + kNormalFloats;
  static_assert(kColor0Float + kColor0Floats <= kNormalFloat &&
                kNormalFloat + kNormalFloats <= gxruntime::gxcore::kVertexFloats);

  struct Entry {
    std::vector<float> out; // the vertices made: the draw's own but for colour 0 and the normal
    std::vector<float> own; // the draw's own colour 0 and normal, kOwnFloats a vertex
    std::vector<uint16_t> indices;
    Inputs inputs{};
    uint64_t used = 0;
  };

  // Only picks the entry (same_draw() compares it all): the indices, the size and the vertices' two ends.
  static uint64_t key_of(const std::vector<float>& vertices, const std::vector<uint16_t>& indices) {
    const size_t bytes = vertices.size() * sizeof(float);
    const size_t ends = std::min<size_t>(bytes, 1024);
    const auto* data = reinterpret_cast<const uint8_t*>(vertices.data());
    return XXH3_64bits_withSeed(indices.data(), indices.size() * sizeof(uint16_t), bytes) ^
           XXH3_64bits(data, ends) * 0x9E3779B97F4A7C15ull ^ XXH3_64bits(data + bytes - ends, ends);
  }

  // Whether `vertices` and `indices` are the draw `entry` was made from.
  static bool same_draw(const Entry& entry, const std::vector<float>& vertices, const std::vector<uint16_t>& indices) {
    constexpr size_t F = gxruntime::gxcore::kVertexFloats;
    if (entry.out.size() != vertices.size() || entry.indices.size() != indices.size() ||
        std::memcmp(entry.indices.data(), indices.data(), indices.size() * sizeof(uint16_t)) != 0)
      return false;
    const size_t count = vertices.size() / F;
    const float* in = vertices.data();
    const float* out = entry.out.data();
    const float* own = entry.own.data();
    constexpr size_t afterColor0 = kColor0Float + kColor0Floats, afterNormal = kNormalFloat + kNormalFloats;
    for (size_t v = 0; v < count; ++v, in += F, out += F, own += kOwnFloats) {
      if (std::memcmp(in, out, kColor0Float * sizeof(float)) != 0 ||
          std::memcmp(in + afterColor0, out + afterColor0, (kNormalFloat - afterColor0) * sizeof(float)) != 0 ||
          std::memcmp(in + afterNormal, out + afterNormal, (F - afterNormal) * sizeof(float)) != 0 ||
          std::memcmp(in + kColor0Float, own, kColor0Floats * sizeof(float)) != 0 ||
          std::memcmp(in + kNormalFloat, own + kColor0Floats, kNormalFloats * sizeof(float)) != 0)
        return false;
    }
    return true;
  }

  absl::flat_hash_map<uint64_t, Entry> entries_;
  uint64_t lastSweep_ = 0, draws_ = 0, hits_ = 0;
};

} // namespace aurora::gfx::gxcore
