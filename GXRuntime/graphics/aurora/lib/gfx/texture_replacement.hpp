#pragma once

#include "texture.hpp"
#include <optional>

namespace aurora::gfx::texture_replacement {
void initialize() noexcept;
void shutdown() noexcept;
void register_tlut(const GXTlutObj* obj, const void* data, GXTlutFmt format, uint16_t entries) noexcept;
void load_tlut(const GXTlutObj* obj, uint32_t idx) noexcept;
std::optional<TextureHandle> find_replacement(const GXTexObj_& obj) noexcept;
std::optional<TextureHandle> find_replacement(const GXTexObj_& obj, const GXTlutObj_& tlut) noexcept;
std::string build_texture_replacement_name(const GXTexObj_& obj) noexcept;
// For renderers that decode guest textures themselves (GXRuntime's gxcore
// path): the Dolphin-named replacement for these guest bytes, if one is
// registered. data holds the texture (at least its base level); tlut the
// palette bytes for C4/C8/C14X2 (tlut_bytes = entries * 2).
std::optional<TextureHandle> find_replacement_for_guest(uint32_t width, uint32_t height, uint32_t format,
                                                        const uint8_t* data, uint32_t data_size,
                                                        const uint8_t* tlut, uint32_t tlut_bytes) noexcept;
bool has_source_replacements() noexcept;
} // namespace aurora::gfx::texture_replacement
