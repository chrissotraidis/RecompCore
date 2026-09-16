// Copyright 2026 GalaxyPad contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Batched equivalent of Dolphin's indexed software vertex converters.
#pragma once

#include <cstring>

namespace IndexedVertexBatch
{
inline u16 ReadIndex(const u8*& p)
{
  const u16 value = (u16(p[0]) << 8) | p[1];
  p += 2;
  return value;
}

inline s16 ReadSigned(const u8* p)
{
  const u16 bits = (u16(p[0]) << 8) | p[1];
  s16 result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <typename T>
inline void Write(u8*& p, T value)
{
  std::memcpy(p, &value, sizeof(value));
  p += sizeof(value);
}

// The caller verifies the entire pipeline, including the terminal skip stage.
// No class layout, guest data, precision policy or output layout changes.
template <bool Matrix, bool Color>
int Run(VertexLoader* loader, const u8* src, u8* dst, int count, int texcoords)
{
  const auto& bases = VertexLoaderManager::cached_arraybases;
  const auto& strides = g_main_cp_state.array_strides;
  const u8* pos_base = bases[CPArray::Position];
  const u8* norm_base = bases[CPArray::Normal];
  const auto pos_stride = strides[CPArray::Position];
  const auto norm_stride = strides[CPArray::Normal];
  loader->m_numLoadedVertices += count;
  loader->m_skippedVertices = 0;
  for (loader->m_remaining = count - 1; loader->m_remaining >= 0; --loader->m_remaining)
  {
    loader->m_tcIndex = 0;
    loader->m_colIndex = 0;
    loader->m_texmtxwrite = loader->m_texmtxread = 0;
    if constexpr (Matrix)
    {
      const u32 matrix = *src++ & 0x3f;
      if (loader->m_remaining < 3)
        VertexLoaderManager::position_matrix_index_cache[loader->m_remaining] = matrix;
      Write(dst, matrix);
    }
    const u16 position = ReadIndex(src);
    loader->m_vertexSkip = position == 0xffff;
    const u8* p = pos_base + position * pos_stride;
    for (int i = 0; i < 3; ++i)
    {
      const float value = ReadSigned(p + 2 * i) * loader->m_posScale;
      if (loader->m_remaining < 3 && !loader->m_vertexSkip)
        VertexLoaderManager::position_cache[loader->m_remaining][i] = value;
      Write(dst, value);
    }
    p = norm_base + ReadIndex(src) * norm_stride;
    for (int i = 0; i < 3; ++i)
    {
      const float value = ReadSigned(p + 2 * i) / float(1u << 14);
      if (loader->m_remaining == 0)
        VertexLoaderManager::normal_cache[i] = value;
      Write(dst, value);
    }
    if constexpr (Color)
    {
      p = bases[CPArray::Color0] + ReadIndex(src) * strides[CPArray::Color0];
      std::memcpy(dst, p, 4);
      dst += 4;
      loader->m_colIndex = 1;
    }
    for (int tc = 0; tc < texcoords; ++tc)
    {
      p = bases[CPArray::TexCoord0 + tc] + ReadIndex(src) * strides[CPArray::TexCoord0 + tc];
      for (int i = 0; i < 2; ++i)
        Write(dst, ReadSigned(p + 2 * i) * loader->m_tcScale[tc]);
      ++loader->m_tcIndex;
    }
    if (loader->m_vertexSkip)
    {
      dst -= loader->m_native_vtx_decl.stride;
      ++loader->m_skippedVertices;
    }
  }
  g_video_buffer_read_ptr = src;
  g_vertex_manager_write_ptr = dst;
  return count - loader->m_skippedVertices;
}
}  // namespace IndexedVertexBatch
