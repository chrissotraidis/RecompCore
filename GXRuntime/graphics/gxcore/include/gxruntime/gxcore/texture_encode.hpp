// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <span>
#include <vector>

namespace gxruntime::gxcore {
// Encode an already format-converted EFB resolve (RGBA8) into guest tiles.
// I8 uses the converted intensity channel, without a second luma conversion.
// Sample render-scaled copies at logical pixel centers. Empty on invalid input
// or unsupported format. Currently supports the Picto Box's I8 and RGB565.
std::vector<std::uint8_t> encode_efb_copy(std::uint32_t format,
    std::uint32_t width, std::uint32_t height,
    std::span<const std::uint8_t> rgba,
    std::uint32_t render_width, std::uint32_t render_height);

// An EFB copy's destination in guest memory: its address, size in bytes, GX
// format and size in texels.
struct EfbCopyExtent {
  std::uint32_t address = 0;
  std::uint32_t byte_size = 0;
  std::uint32_t format = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};
// The row of `earlier`'s buffer at which `later` starts, when `later` lies
// inside that buffer past its start, with its format and width, at a whole
// row: in memory it rewrites those rows (Wind Waker's drawDepth copies a frame
// whose viewport starts above the screen that many rows into its buffer).
// -1 otherwise.
constexpr std::int64_t efb_copy_row_within(const EfbCopyExtent& earlier, const EfbCopyExtent& later) {
  if (later.address <= earlier.address || later.format != earlier.format || later.width != earlier.width ||
      earlier.height == 0u || later.byte_size == 0u || earlier.byte_size % earlier.height != 0u)
    return -1;
  const std::uint64_t offset = later.address - earlier.address;
  const std::uint64_t row_bytes = earlier.byte_size / earlier.height;
  if (row_bytes == 0u || offset % row_bytes != 0u || offset + later.byte_size > earlier.byte_size)
    return -1;
  return static_cast<std::int64_t>(offset / row_bytes);
}
}
