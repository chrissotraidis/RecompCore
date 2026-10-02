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
}
