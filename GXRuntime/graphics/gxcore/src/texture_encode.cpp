// SPDX-License-Identifier: GPL-3.0-or-later
#include "gxruntime/gxcore/texture_encode.hpp"
#include <algorithm>

namespace gxruntime::gxcore {
std::vector<std::uint8_t> encode_efb_copy(std::uint32_t format,
    std::uint32_t width, std::uint32_t height,
    std::span<const std::uint8_t> rgba,
    std::uint32_t render_width, std::uint32_t render_height) {
  if ((format != 1u && format != 4u) || width == 0 || height == 0 ||
      render_width == 0 || render_height == 0 ||
      std::uint64_t(render_width) * render_height > rgba.size() / 4u)
    return {};
  const std::uint64_t block_width = format == 1u ? 8u : 4u;
  const std::uint64_t blocks_x = (std::uint64_t(width) + block_width - 1) / block_width;
  const std::uint64_t blocks_y = (std::uint64_t(height) + 3) / 4;
  // Reject oversized untrusted dimensions before allocating or indexing.
  if (blocks_x * blocks_y > 0x02000000u / 32u)
    return {};
  std::vector<std::uint8_t> out(blocks_x * blocks_y * 32u);
  std::size_t offset = 0;
  for (std::uint64_t by = 0; by < blocks_y; ++by) {
    for (std::uint64_t bx = 0; bx < blocks_x; ++bx) {
      for (std::uint32_t y = 0; y < 4u; ++y) {
        for (std::uint32_t x = 0; x < block_width; ++x) {
          const auto lx = std::min(bx * block_width + x, std::uint64_t(width - 1));
          const auto ly = std::min(by * 4u + y, std::uint64_t(height - 1));
          const auto sx = ((lx * 2u + 1u) * render_width) / (std::uint64_t(width) * 2u);
          const auto sy = ((ly * 2u + 1u) * render_height) / (std::uint64_t(height) * 2u);
          const auto* pixel = rgba.data() + (sy * render_width + sx) * 4u;
          if (format == 1u) {
            out[offset++] = pixel[0];
          } else {
            const auto rgb565 = std::uint16_t((pixel[0] >> 3u) << 11u |
                (pixel[1] >> 2u) << 5u | (pixel[2] >> 3u));
            out[offset++] = std::uint8_t(rgb565 >> 8u);
            out[offset++] = std::uint8_t(rgb565);
          }
        }
      }
    }
  }
  return out;
}
}
