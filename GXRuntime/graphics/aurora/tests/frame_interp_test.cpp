// Standalone checks for lib/gfx/frame_interp.cpp: draw matching across
// frames, plausibility rejection, the halfway blend and the cut verdict.
#include "../lib/gfx/frame_interp.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace fi = aurora::gfx::frame_interp;
namespace gxc = gxruntime::gxcore;

static int g_failures = 0;
#define CHECK(cond)                                                                                                    \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond);                                             \
      ++g_failures;                                                                                                    \
    }                                                                                                                  \
  } while (0)

static bool near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

// A perspective draw at view-space (x, y, z), rotated by `yaw` degrees.
static gxc::VertexShaderConstants draw_at(float x, float y, float z, float yaw = 0.f) {
  gxc::VertexShaderConstants c{};
  const float r = yaw * 3.14159265f / 180.f;
  const float cs = std::cos(r), sn = std::sin(r);
  const float m[3][4] = {{cs, 0, sn, x}, {0, 1, 0, y}, {-sn, 0, cs, z}};
  std::memcpy(c.posnormalmatrix, m, sizeof(m));
  const float p[4][4] = {{1.5f, 0, 0, 0}, {0, 2.f, 0, 0}, {0, 0, -1.f, -10.f}, {0, 0, -1.f, 0}};
  std::memcpy(c.projection, p, sizeof(p));
  for (int i = 0; i + 2 < 64; i += 3) {
    c.transformmatrices[i][0] = c.transformmatrices[i + 1][1] = c.transformmatrices[i + 2][2] = 1.f;
  }
  return c;
}

// A world-space model at (x, y, z) seen by a camera at (cx, 0, cz) turned
// `yaw` degrees: the draw's matrix is the view matrix times the model's.
static gxc::VertexShaderConstants seen(float yaw, float cx, float cz, float x, float y, float z) {
  const float r = yaw * 3.14159265f / 180.f;
  const float cs = std::cos(r), sn = std::sin(r);
  const float dx = x - cx, dz = z - cz;
  return draw_at(cs * dx + sn * dz, y, -sn * dx + cs * dz, yaw);
}

struct Camera {
  float yaw, x, z;
};

// Rooms (a key each) that vote for the camera's motion, then a list of grass
// clumps (one key) at the given world x positions, the one at `liftedX` raised
// by `lift`. Each clump's blended constants, if it was blended, go to `out`
// (blend_draw's result is only valid until its next call).
struct Result {
  bool blended = false;
  gxc::VertexShaderConstants constants;
};
static void draw_scene(const Camera& c, const float* clumps, int count, Result* out, float liftedX = 0.f,
                       float lift = 0.f) {
  for (int room = 0; room < 6; ++room)
    fi::blend_draw(200 + room, 0, seen(c.yaw, c.x, c.z, room * 300.f - 750.f, -50.f, -900.f - room * 40.f));
  for (int i = 0; i < count; ++i) {
    const float y = clumps[i] == liftedX ? lift : 0.f;
    const auto* result = fi::blend_draw(500, 0, seen(c.yaw, c.x, c.z, clumps[i], y, -800.f));
    out[i].blended = result != nullptr;
    if (result != nullptr)
      out[i].constants = *result;
  }
}

// Whether a clump at world x was drawn halfway between its own two places
// (raised by yBefore and y).
static bool halfway(const Result& result, const Camera& before, const Camera& now, float x, float yBefore = 0.f,
                    float y = 0.f) {
  if (!result.blended)
    return false;
  const auto a = seen(before.yaw, before.x, before.z, x, yBefore, -800.f);
  const auto b = seen(now.yaw, now.x, now.z, x, y, -800.f);
  for (int r = 0; r < 3; ++r) {
    const float expected = (a.posnormalmatrix[r][3] + b.posnormalmatrix[r][3]) / 2.f;
    if (std::fabs(result.constants.posnormalmatrix[r][3] - expected) > 0.01f)
      return false;
  }
  return true;
}

int main() {
  fi::set_enabled(true);
  const uint64_t kShip = 11, kTree = 22, kHud = 33;

  // Frame 1: the ship, two trees, the HUD.
  CHECK(fi::blend_draw(kShip, 0, draw_at(0, 0, -500)) == nullptr); // no previous frame yet
  fi::blend_draw(kTree, 0, draw_at(-200, 0, -800));
  fi::blend_draw(kTree, 0, draw_at(300, 0, -900));
  fi::blend_draw(kHud, 0, draw_at(10, 10, -1));
  CHECK(!fi::frame_verdict());
  fi::end_game_frame();

  // Frame 2: the ship moved 20 units and turned 10 degrees; the trees are in
  // the other order; the HUD did not move.
  const auto* ship = fi::blend_draw(kShip, 0, draw_at(20, 0, -500, 10.f));
  CHECK(ship != nullptr);
  if (ship != nullptr) {
    CHECK(near(ship->posnormalmatrix[0][3], 10.f));
    CHECK(near(ship->posnormalmatrix[2][3], -500.f));
    CHECK(near(ship->posnormalmatrix[0][0], (1.f + std::cos(10.f * 3.14159265f / 180.f)) / 2.f));
  }
  const auto* treeA = fi::blend_draw(kTree, 0, draw_at(305, 0, -900));
  CHECK(treeA != nullptr);
  if (treeA != nullptr) // occurrence 0 was the tree at -200, 505 away: implausible, so the nearest
    CHECK(near(treeA->posnormalmatrix[0][3], 302.5f));
  const auto* treeB = fi::blend_draw(kTree, 0, draw_at(-195, 0, -800));
  CHECK(treeB != nullptr);
  if (treeB != nullptr)
    CHECK(near(treeB->posnormalmatrix[0][3], -197.5f));
  CHECK(fi::blend_draw(kHud, 0, draw_at(10, 10, -1)) == nullptr); // identical: its own constants
  CHECK(fi::frame_verdict());
  fi::end_game_frame();

  // Frame 3: a camera cut. Everything turns 120 degrees and jumps.
  const float cutX[] = {900, -700, 400, -300, 800, -600, 500, -400, 700, -500, 600, -800, 300, -900, 200, -100};
  fi::blend_draw(kShip, 0, draw_at(cutX[0], 0, 200, 130.f));
  for (int i = 0; i < 20; ++i)
    fi::blend_draw(kTree, 0, draw_at(cutX[i % 16], 50, 300, 120.f));
  CHECK(!fi::frame_verdict());
  fi::end_game_frame();

  // Frame 4: a scrolling texture wraps (0.98 -> 0.02) and keeps its value;
  // one that moves a little is blended.
  {
    auto a = draw_at(0, 0, -300);
    a.texmatrices[0][0] = a.texmatrices[1][1] = 1.f;
    a.texmatrices[0][3] = 0.98f;
    a.texmatrices[3][0] = a.texmatrices[4][1] = 1.f;
    a.texmatrices[3][3] = 0.10f;
    fi::blend_draw(99, 0, a);
    fi::end_game_frame();
    auto b = a;
    b.texmatrices[0][3] = 0.02f;
    b.texmatrices[3][3] = 0.20f;
    b.posnormalmatrix[0][3] = 4.f;
    const auto* blended = fi::blend_draw(99, 0, b);
    CHECK(blended != nullptr);
    if (blended != nullptr) {
      CHECK(near(blended->texmatrices[0][3], 0.02f));
      CHECK(near(blended->texmatrices[3][3], 0.15f));
      CHECK(near(blended->posnormalmatrix[0][3], 2.f));
    }
    fi::end_game_frame();
  }

  // A collapsed (hidden) object is not grown halfway.
  {
    auto shown = draw_at(0, 0, -300);
    fi::blend_draw(77, 0, shown);
    fi::end_game_frame();
    auto hidden = shown;
    std::memset(hidden.posnormalmatrix, 0, sizeof(float) * 12);
    CHECK(fi::blend_draw(77, 0, hidden) == nullptr);
    fi::end_game_frame();
  }

  // Grass clumps (copies of one draw) under a turning camera. The camera's
  // motion, voted for by the rooms, carries each clump to its own place in
  // the frame before, whatever the list order: a clump culled at the front
  // shifts every later one onto its neighbour, 40 units away (well inside the
  // plausibility bound), and the turn moves distant clumps farther than that.
  {
    const Camera a{0.f, 0.f, 0.f}, b{6.f, 10.f, -20.f}, c{12.f, 15.f, -45.f}, d{18.f, 20.f, -70.f},
        e{24.f, 25.f, -95.f}, f{30.f, 30.f, -120.f};
    Result r[6];
    const float first[] = {-100, -60, -20, 20, 60, 100};
    draw_scene(a, first, 6, r);
    fi::end_game_frame();

    // The first clump left the view; the others still pair with themselves.
    const float second[] = {-60, -20, 20, 60, 100};
    draw_scene(b, second, 5, r);
    for (int i = 0; i < 5; ++i)
      CHECK(halfway(r[i], a, b, second[i]));
    CHECK(fi::frame_verdict());
    fi::end_game_frame();

    // A clump comes into view at the end of the list: nothing stood where it
    // is, so it is drawn where the in-between camera sees it (blended from
    // where the camera's motion carries it from), not slid from a neighbour.
    const float third[] = {-60, -20, 20, 60, 100, 140};
    draw_scene(c, third, 6, r);
    CHECK(halfway(r[5], b, c, 140));
    fi::end_game_frame();
    draw_scene(d, third, 6, r);
    CHECK(halfway(r[5], c, d, 140));
    fi::end_game_frame();

    // A clump that starts to move at an actor's pace (a bush Link lifts) is
    // taken for one that stood at its new place for a frame, then pairs with
    // itself.
    draw_scene(e, third, 6, r, 60.f, 15.f);
    CHECK(halfway(r[3], d, e, 60, 15.f, 15.f) && halfway(r[4], d, e, 100));
    fi::end_game_frame();
    draw_scene(f, third, 6, r, 60.f, 30.f);
    CHECK(halfway(r[3], e, f, 60, 15.f, 30.f));
    fi::end_game_frame();
  }

  // Draw keys: the same payload gives the same key; another primitive does not.
  {
    const uint8_t payload[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    gxc::DrawPlan plan;
    plan.match_payload = payload;
    plan.match_payload_size = sizeof(payload);
    plan.match_primitive = 0x90;
    plan.vertex_count = 3;
    const uint64_t k1 = fi::draw_key(plan);
    CHECK(k1 != 0 && k1 == fi::draw_key(plan));
    plan.match_primitive = 0x98;
    CHECK(fi::draw_key(plan) != k1);
    plan.match_payload = nullptr;
    CHECK(fi::draw_key(plan) == 0);
  }

  if (g_failures == 0)
    std::puts("frame_interp_test: all checks passed");
  return g_failures == 0 ? 0 : 1;
}
