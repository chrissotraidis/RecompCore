// SPDX-License-Identifier: GPL-3.0-or-later
// cutout_alpha_texmap(): the texmap an alpha-tested draw's cutout comes from,
// for the HD look's sun map casters (aurora gxcore_draw.cpp submit_draw_plan).

#include "gxruntime/gxcore/shader.hpp"

#include <cstdio>

namespace gxc = gxruntime::gxcore;

static int g_failures = 0;

#define CHECK(cond)                                                           \
  do {                                                                        \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
      ++g_failures;                                                           \
    }                                                                         \
  } while (0)

namespace {

constexpr std::uint8_t kZero = 7, kKonst = 6, kTexA = 4, kAPrev = 0;

gxc::TexGenKey texgen(gxc::TexGenType type, gxc::TexSourceRow row) {
  gxc::TexGenKey g{};
  g.enabled = 1;
  g.texgentype = static_cast<std::uint8_t>(type);
  g.sourcerow = static_cast<std::uint8_t>(row);
  return g;
}

gxc::TevStageKey stage(std::uint8_t texcoord, std::uint8_t texmap, bool enable, std::uint8_t a, std::uint8_t b,
                       std::uint8_t c, std::uint8_t d) {
  gxc::TevStageKey s{};
  s.tevorders_texcoord = texcoord;
  s.tevorders_texmap = texmap;
  s.tevorders_enable = enable ? 1 : 0;
  s.ac_a = a;
  s.ac_b = b;
  s.ac_c = c;
  s.ac_d = d;
  return s;
}

// The Windfall girl's body as dumped from the game (s6248): texgen 0 regular
// from TEX0, texgen 1 the toon ramp from colour 0 (SRTG); stage 0 the ramp
// (texmap 1, no alpha read), stage 1 konst x TEXA from texmap 0, stage 2 APREV.
gxc::ShaderKey windfall_girl() {
  gxc::ShaderKey k{};
  k.textured = 1;
  k.tev_valid = 1;
  k.num_tex_gens = 2;
  k.tex_gens[0] = texgen(gxc::TexGenType::Regular, gxc::TexSourceRow::Tex0);
  k.tex_gens[1] = texgen(gxc::TexGenType::Color0, gxc::TexSourceRow::Colors);
  k.num_tev_stages = 3;
  k.tev_stages[0] = stage(1, 1, true, kZero, kZero, kZero, kZero);
  k.tev_stages[1] = stage(0, 0, true, kZero, kKonst, kTexA, kZero);
  k.tev_stages[2] = stage(1, 1, true, kAPrev, kZero, kZero, kZero);
  k.alpha_comp0 = 6; // GEQUAL
  k.alpha_comp1 = 3; // LEQUAL
  k.alpha_logic = 0; // AND
  return k;
}

void test_girl_takes_texmap0() { CHECK(gxc::cutout_alpha_texmap(windfall_girl()) == 0); }

void test_alpha_from_second_texmap() {
  // The same, with the cutout texture on texmap 2.
  gxc::ShaderKey k = windfall_girl();
  k.tev_stages[1].tevorders_texmap = 2;
  CHECK(gxc::cutout_alpha_texmap(k) == 2);
}

void test_two_alpha_textures_refused() {
  gxc::ShaderKey k = windfall_girl();
  k.tev_stages[2] = stage(0, 1, true, kAPrev, kZero, kTexA, kZero);
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
}

void test_same_texture_twice_kept() {
  gxc::ShaderKey k = windfall_girl();
  k.tev_stages[2] = stage(0, 0, true, kAPrev, kZero, kTexA, kZero);
  CHECK(gxc::cutout_alpha_texmap(k) == 0);
}

void test_alpha_not_from_texture_refused() {
  // Alpha from konst and the raster only: no texture to cut by.
  gxc::ShaderKey k = windfall_girl();
  k.tev_stages[1].ac_c = kZero;
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
}

void test_texcoord_not_from_tex0_refused() {
  // The alpha texture's coordinate from TEX1 (a caster carries TEX0 only) or a
  // non-regular texgen.
  gxc::ShaderKey k = windfall_girl();
  k.tex_gens[0].sourcerow = static_cast<std::uint8_t>(gxc::TexSourceRow::Tex0) + 1;
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
  k = windfall_girl();
  k.tev_stages[1].tevorders_texcoord = 1; // the ramp's SRTG coordinate
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
  k = windfall_girl();
  k.tex_gens[0].projection = 1; // STQ
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
}

void test_texcoord_out_of_range_refused() {
  gxc::ShaderKey k = windfall_girl();
  k.num_tex_gens = 0;
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
}

void test_disabled_texture_stage_refused() {
  gxc::ShaderKey k = windfall_girl();
  k.tev_stages[1].tevorders_enable = 0;
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
}

void test_indirect_warp_refused() {
  gxc::ShaderKey k = windfall_girl();
  k.num_ind_stages = 1;
  k.tev_stages[1].ind_stage = 0;
  k.tev_stages[1].ind_matrix_index = 1;
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
}

void test_untextured_and_passthrough() {
  gxc::ShaderKey k{};
  CHECK(gxc::cutout_alpha_texmap(k) == -1);
  k.textured = 1; // no TEV: texmap 0 through texcoord 0
  k.num_tex_gens = 1;
  k.tex_gens[0] = texgen(gxc::TexGenType::Regular, gxc::TexSourceRow::Tex0);
  CHECK(gxc::cutout_alpha_texmap(k) == 0);
}

} // namespace

int main() {
  test_girl_takes_texmap0();
  test_alpha_from_second_texmap();
  test_two_alpha_textures_refused();
  test_same_texture_twice_kept();
  test_alpha_not_from_texture_refused();
  test_texcoord_not_from_tex0_refused();
  test_texcoord_out_of_range_refused();
  test_disabled_texture_stage_refused();
  test_indirect_warp_refused();
  test_untextured_and_passthrough();
  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("gxcore_cutout_tests: ok\n");
  return 0;
}
