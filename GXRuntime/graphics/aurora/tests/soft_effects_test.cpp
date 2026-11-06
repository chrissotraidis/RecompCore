// Efficient soft effects (soft_effects.hpp): which draws take the reduced-size
// buffer, which runs of them do, and that the buffer blended over the EFB once
// equals each quad blended onto the EFB in turn.

#include "../lib/gfx/soft_effects.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace se = aurora::gfx::soft_effects;
namespace gxc = gxruntime::gxcore;

static int g_failures = 0;
#define CHECK(cond)                                                                                                    \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);                                    \
      ++g_failures;                                                                                                    \
    }                                                                                                                  \
  } while (0)

// A pipeline key as the draw logs describe one: TEV stages, indirect stages,
// blending (enable, source, destination), depth test and write, colour and
// alpha writes.
static gxc::PipelineKey key(unsigned tev, unsigned ind, unsigned blend, unsigned src, unsigned dst,
                            unsigned depthTest, unsigned depthWrite, unsigned colorWrite = 1, unsigned alphaWrite = 1) {
  gxc::PipelineKey k{};
  k.shader.num_tev_stages = static_cast<std::uint8_t>(tev);
  k.shader.num_ind_stages = static_cast<std::uint8_t>(ind);
  k.blend_enable = static_cast<std::uint8_t>(blend);
  k.src_factor = static_cast<std::uint8_t>(src);
  k.dst_factor = static_cast<std::uint8_t>(dst);
  k.depth_test = static_cast<std::uint8_t>(depthTest);
  k.depth_func = 3;
  k.depth_update = static_cast<std::uint8_t>(depthWrite);
  k.color_update = static_cast<std::uint8_t>(colorWrite);
  k.alpha_update = static_cast<std::uint8_t>(alphaWrite);
  return k;
}

struct Rgba {
  double r, g, b, a;
};

// One quad blended onto a target: colour (SrcAlpha, OneMinusSrcAlpha).
static Rgba over(Rgba dst, Rgba src) {
  return {src.r * src.a + dst.r * (1 - src.a), src.g * src.a + dst.g * (1 - src.a),
          src.b * src.a + dst.b * (1 - src.a), dst.a};
}

int main() {
  // The Lava Room's heat haze and the Shipwreck's veil (draw logs, scale 3):
  // tev=1 ind=0 blend=1(4,5) z=test0/write0, quads of 6 indices.
  const auto haze = key(1, 0, 1, 4, 5, 0, 0);
  CHECK(se::qualifies_key(haze));
  CHECK(se::qualifies_size(6, 1));
  CHECK(se::qualifies_size(12, 1));
  CHECK(se::qualifies_size(24, 2)); // two quads merged into one batched draw
  CHECK(!se::qualifies_size(18, 1));
  CHECK(!se::qualifies_size(0, 1));
  CHECK(!se::qualifies_size(6, 0));

  // What must keep the EFB:
  CHECK(!se::qualifies_key(key(2, 0, 0, 1, 0, 1, 1)));          // opaque scenery (Lava Room 50-99)
  CHECK(!se::qualifies_key(key(1, 0, 1, 4, 5, 1, 0)));          // the same blend, depth tested (Shipwreck)
  CHECK(!se::qualifies_key(key(1, 0, 1, 1, 1, 0, 0)));          // additive
  CHECK(!se::qualifies_key(key(1, 0, 1, 6, 7, 0, 0)));          // destination alpha (Outset)
  CHECK(!se::qualifies_key(key(1, 0, 1, 6, 0, 0, 0, 0, 1)));    // alpha only (Outset)
  CHECK(!se::qualifies_key(key(3, 1, 1, 4, 5, 1, 1)));          // indirect, three stages (Lava Room 650-699)
  CHECK(!se::qualifies_key(key(2, 0, 1, 4, 5, 0, 0)));          // two TEV stages
  CHECK(!se::qualifies_key(key(1, 0, 0, 4, 5, 0, 0)));          // not blended
  CHECK(!se::qualifies_key(key(1, 0, 1, 4, 5, 0, 0, 0, 1)));    // colour not written
  {
    auto k = haze;
    k.blend_subtract = 1;
    CHECK(!se::qualifies_key(k));
    k = haze;
    k.primitive_topology = 1; // lines
    CHECK(!se::qualifies_key(k));
    k = haze;
    k.shader.use_dst_alpha = 1;
    CHECK(!se::qualifies_key(k));
  }

  // Runs: eight or more in a row; a draw that does not qualify, or a change of
  // alpha writing, ends one.
  {
    std::vector<bool> soft, alpha;
    const auto add = [&](int n, bool s, bool a = true) {
      for (int i = 0; i < n; ++i) {
        soft.push_back(s);
        alpha.push_back(a);
      }
    };
    add(50, false);       // the scene
    add(81, true);        // the haze, 50-130
    add(3, false);        // the HUD
    add(7, true);         // too few
    add(1, false);
    add(10, true, true);  // 142-151
    add(9, true, false);  // 152-160: alpha writing changes, a run of its own
    const auto runs = se::find_runs(soft, alpha);
    CHECK(runs.size() == 3);
    if (runs.size() == 3) {
      CHECK(runs[0].first == 50 && runs[0].count == 81);
      CHECK(runs[1].first == 142 && runs[1].count == 10);
      CHECK(runs[2].first == 152 && runs[2].count == 9);
    }
    CHECK(se::find_runs({}, {}).empty());
    CHECK(se::find_runs(std::vector<bool>(7, true), std::vector<bool>(7, true)).empty());
    CHECK(se::find_runs(std::vector<bool>(8, true), std::vector<bool>(8, true)).size() == 1);
  }

  // The buffer, from (0,0,0,0) with colour (SrcAlpha, OneMinusSrcAlpha) and
  // alpha (One, OneMinusSrcAlpha), blended over the EFB with (One,
  // OneMinusSrcAlpha): the same colour as each quad blended in turn, for any
  // number of layers and any coverage.
  {
    std::mt19937 random(504);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    double worst = 0.0;
    for (int trial = 0; trial < 2000; ++trial) {
      const Rgba efb{unit(random), unit(random), unit(random), unit(random)};
      const int layers = 1 + trial % 90;
      Rgba direct = efb;
      Rgba buffer{0, 0, 0, 0};
      for (int i = 0; i < layers; ++i) {
        const Rgba quad{unit(random), unit(random), unit(random), unit(random) * (trial % 3 == 0 ? 0.05 : 1.0)};
        direct = over(direct, quad);
        buffer = {quad.r * quad.a + buffer.r * (1 - quad.a), quad.g * quad.a + buffer.g * (1 - quad.a),
                  quad.b * quad.a + buffer.b * (1 - quad.a), quad.a + buffer.a * (1 - quad.a)};
      }
      const Rgba composite{buffer.r + efb.r * (1 - buffer.a), buffer.g + efb.g * (1 - buffer.a),
                           buffer.b + efb.b * (1 - buffer.a), efb.a};
      worst = std::max({worst, std::fabs(composite.r - direct.r), std::fabs(composite.g - direct.g),
                        std::fabs(composite.b - direct.b)});
    }
    CHECK(worst < 1e-9);
  }

  if (g_failures == 0)
    std::puts("soft_effects_test: all checks passed");
  return g_failures == 0 ? 0 : 1;
}
