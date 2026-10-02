// SPDX-License-Identifier: GPL-3.0-or-later
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "gxruntime/gxcore/texture_encode.hpp"
#include "gxruntime/gxcore/texture_decode.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
using namespace gxruntime::gxcore;

int main() {
  std::vector<std::uint8_t> rgba(16 * 8 * 4);
  for (unsigned y = 0; y < 8; ++y)
    for (unsigned x = 0; x < 16; ++x)
      for (unsigned c = 0; c < 4; ++c)
        rgba[(y * 16 + x) * 4 + c] = y * 16 + x;
  const auto i8 = encode_efb_copy(1, 16, 8, rgba, 16, 8);
  assert(i8.size() == 128);
  // Four 8x4 blocks, in block-row order, with no luma clamping/conversion.
  assert(i8[0] == 0 && i8[7] == 7 && i8[8] == 16 && i8[31] == 55);
  assert(i8[32] == 8 && i8[63] == 63 && i8[64] == 64 && i8[127] == 127);
  const auto decoded = decode_texture(1, 16, 8, i8.data(), i8.size());
  for (unsigned p = 0; p < 128; ++p)
    assert(decoded[p * 4] == rgba[p * 4]);

  const std::uint8_t colors[4][4] = {
      {255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 255, 255}};
  rgba.resize(8 * 4 * 4);
  for (unsigned p = 0; p < 32; ++p)
    for (unsigned c = 0; c < 4; ++c)
      rgba[p * 4 + c] = colors[p % 4][c];
  const auto rgb = encode_efb_copy(4, 8, 4, rgba, 8, 4);
  assert(rgb.size() == 64);
  const std::uint8_t words[] = {0xf8,0x00, 0x07,0xe0, 0x00,0x1f, 0xff,0xff};
  for (unsigned p = 0; p < 64; ++p)
    assert(rgb[p] == words[p % 8]);
  const auto rgb_decoded = decode_texture(4, 8, 4, rgb.data(), rgb.size());
  assert(rgb_decoded == rgba);

  // Logical pixel centers in a 3x render-scale resolve.
  rgba.assign(6 * 3 * 4, 0);
  rgba[(1 * 6 + 1) * 4] = 42;
  rgba[(1 * 6 + 4) * 4] = 198;
  const auto scaled = encode_efb_copy(1, 2, 1, rgba, 6, 3);
  assert(scaled.size() == 32 && scaled[0] == 42 && scaled[1] == 198);
  assert(scaled[7] == 198 && scaled[8] == 42 && scaled[31] == 198);
  assert(encode_efb_copy(6, 2, 1, rgba, 6, 3).empty());
  assert(encode_efb_copy(1, 0, 1, rgba, 6, 3).empty());
  assert(encode_efb_copy(1, 2, 1, rgba, 7, 3).empty());
  assert(encode_efb_copy(1, UINT32_MAX, UINT32_MAX, rgba, 6, 3).empty());
}
