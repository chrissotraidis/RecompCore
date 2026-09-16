// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/VertexLoader.h"

#include "Common/Assert.h"
#include "Common/CommonTypes.h"

#include "VideoCommon/VertexLoaderManager.h"
#include "VideoCommon/VertexLoaderUtils.h"
#include "VideoCommon/VertexLoader_Color.h"
#include "VideoCommon/VertexLoader_Normal.h"
#include "VideoCommon/VertexLoader_Position.h"
#include "VideoCommon/VertexLoader_TextCoord.h"
#include "VideoCommon/VideoCommon.h"

#ifndef GALAXYPAD_INDEXED_VERTEX_BATCH
#define GALAXYPAD_INDEXED_VERTEX_BATCH 0
#endif
#if GALAXYPAD_INDEXED_VERTEX_BATCH
#include "VideoCommon/VertexLoaderIndexedBatch.h"
#include <cstdio>
#endif

// This pointer is used as the source/dst for all fixed function loader calls
const u8* g_video_buffer_read_ptr;
u8* g_vertex_manager_write_ptr;

static void PosMtx_ReadDirect_UByte(VertexLoader* loader)
{
  u32 posmtx = DataRead<u8>() & 0x3f;
  if (loader->m_remaining < 3)
    VertexLoaderManager::position_matrix_index_cache[loader->m_remaining] = posmtx;
  DataWrite<u32>(posmtx);
  PRIM_LOG("posmtx: {}, ", posmtx);
}

static void TexMtx_ReadDirect_UByte(VertexLoader* loader)
{
  loader->m_curtexmtx[loader->m_texmtxread] = DataRead<u8>() & 0x3f;

  PRIM_LOG("texmtx{}: {}, ", loader->m_texmtxread, loader->m_curtexmtx[loader->m_texmtxread]);
  loader->m_texmtxread++;
}

static void TexMtx_Write_Float(VertexLoader* loader)
{
  DataWrite(float(loader->m_curtexmtx[loader->m_texmtxwrite++]));
}

static void TexMtx_Write_Float2(VertexLoader* loader)
{
  DataWrite(0.f);
  DataWrite(float(loader->m_curtexmtx[loader->m_texmtxwrite++]));
}

static void TexMtx_Write_Float3(VertexLoader* loader)
{
  DataWrite(0.f);
  DataWrite(0.f);
  DataWrite(float(loader->m_curtexmtx[loader->m_texmtxwrite++]));
}

static void SkipVertex(VertexLoader* loader)
{
  if (loader->m_vertexSkip)
  {
    // reset the output buffer
    g_vertex_manager_write_ptr -= loader->m_native_vtx_decl.stride;

    loader->m_skippedVertices++;
  }
}

VertexLoader::VertexLoader(const TVtxDesc& vtx_desc, const VAT& vtx_attr)
    : VertexLoaderBase(vtx_desc, vtx_attr)
{
  CompileVertexTranslator();

  // generate frac factors
  m_posScale = 1.0f / (1U << m_VtxAttr.g0.PosFrac);
  for (u32 i = 0; i < 8; i++)
    m_tcScale[i] = 1.0f / (1U << m_VtxAttr.GetTexFrac(i));
}

void VertexLoader::CompileVertexTranslator()
{
  // Position in pc vertex format.
  int nat_offset = 0;

  // Position Matrix Index
  if (m_VtxDesc.low.PosMatIdx)
  {
    WriteCall(PosMtx_ReadDirect_UByte);
    m_native_vtx_decl.posmtx.components = 4;
    m_native_vtx_decl.posmtx.enable = true;
    m_native_vtx_decl.posmtx.offset = nat_offset;
    m_native_vtx_decl.posmtx.type = ComponentFormat::UByte;
    m_native_vtx_decl.posmtx.integer = true;
    nat_offset += 4;
  }

  for (auto texmtxidx : m_VtxDesc.low.TexMatIdx)
  {
    if (texmtxidx)
      WriteCall(TexMtx_ReadDirect_UByte);
  }

  // Write vertex position loader
  WriteCall(VertexLoader_Position::GetFunction(m_VtxDesc.low.Position, m_VtxAttr.g0.PosFormat,
                                               m_VtxAttr.g0.PosElements));

  int pos_elements = m_VtxAttr.g0.PosElements == CoordComponentCount::XY ? 2 : 3;
  m_native_vtx_decl.position.components = pos_elements;
  m_native_vtx_decl.position.enable = true;
  m_native_vtx_decl.position.offset = nat_offset;
  m_native_vtx_decl.position.type = ComponentFormat::Float;
  m_native_vtx_decl.position.integer = false;
  nat_offset += pos_elements * sizeof(float);

  // Normals
  if (m_VtxDesc.low.Normal != VertexComponentFormat::NotPresent)
  {
    TPipelineFunction pFunc =
        VertexLoader_Normal::GetFunction(m_VtxDesc.low.Normal, m_VtxAttr.g0.NormalFormat,
                                         m_VtxAttr.g0.NormalElements, m_VtxAttr.g0.NormalIndex3);

    if (pFunc == nullptr)
    {
      PanicAlertFmt("VertexLoader_Normal::GetFunction({} {} {} {}) returned zero!",
                    m_VtxDesc.low.Normal, m_VtxAttr.g0.NormalFormat, m_VtxAttr.g0.NormalElements,
                    m_VtxAttr.g0.NormalIndex3);
    }
    WriteCall(pFunc);

    for (int i = 0; i < (m_VtxAttr.g0.NormalElements == NormalComponentCount::NTB ? 3 : 1); i++)
    {
      m_native_vtx_decl.normals[i].components = 3;
      m_native_vtx_decl.normals[i].enable = true;
      m_native_vtx_decl.normals[i].offset = nat_offset;
      m_native_vtx_decl.normals[i].type = ComponentFormat::Float;
      m_native_vtx_decl.normals[i].integer = false;
      nat_offset += 12;
    }
  }

  for (size_t i = 0; i < m_VtxDesc.low.Color.Size(); i++)
  {
    m_native_vtx_decl.colors[i].components = 4;
    m_native_vtx_decl.colors[i].type = ComponentFormat::UByte;
    m_native_vtx_decl.colors[i].integer = false;

    TPipelineFunction pFunc =
        VertexLoader_Color::GetFunction(m_VtxDesc.low.Color[i], m_VtxAttr.GetColorFormat(i));

    if (pFunc != nullptr)
    {
      WriteCall(pFunc);
    }
    else
    {
      ASSERT(m_VtxDesc.low.Color[i] == VertexComponentFormat::NotPresent);
      // Keep colIndex in sync if color 0 is absent but color 1 is present
      if (i == 0 && m_VtxDesc.low.Color[1] != VertexComponentFormat::NotPresent)
        WriteCall(VertexLoader_Color::GetDummyFunction());
    }

    if (m_VtxDesc.low.Color[i] != VertexComponentFormat::NotPresent)
    {
      m_native_vtx_decl.colors[i].offset = nat_offset;
      m_native_vtx_decl.colors[i].enable = true;
      nat_offset += 4;
    }
  }

  // Texture matrix indices (remove if corresponding texture coordinate isn't enabled)
  for (size_t i = 0; i < m_VtxDesc.high.TexCoord.Size(); i++)
  {
    m_native_vtx_decl.texcoords[i].offset = nat_offset;
    m_native_vtx_decl.texcoords[i].type = ComponentFormat::Float;
    m_native_vtx_decl.texcoords[i].integer = false;

    const auto tc = m_VtxDesc.high.TexCoord[i].Value();
    const auto format = m_VtxAttr.GetTexFormat(i);
    const auto elements = m_VtxAttr.GetTexElements(i);

    if (tc != VertexComponentFormat::NotPresent)
    {
      ASSERT_MSG(VIDEO, VertexComponentFormat::Direct <= tc && tc <= VertexComponentFormat::Index16,
                 "Invalid texture coordinates!\n(tc = {})", tc);
      ASSERT_MSG(VIDEO, ComponentFormat::UByte <= format && format <= ComponentFormat::Float,
                 "Invalid texture coordinates format!\n(format = {})", format);
      ASSERT_MSG(VIDEO, elements == TexComponentCount::S || elements == TexComponentCount::ST,
                 "Invalid number of texture coordinates elements!\n(elements = {})", elements);

      WriteCall(VertexLoader_TextCoord::GetFunction(tc, format, elements));
    }

    if (m_VtxDesc.low.TexMatIdx[i])
    {
      m_native_vtx_decl.texcoords[i].enable = true;
      m_native_vtx_decl.texcoords[i].components = 3;
      nat_offset += 12;
      if (tc != VertexComponentFormat::NotPresent)
      {
        // if texmtx is included, texcoord will always be 3 floats, z will be the texmtx index
        WriteCall(elements == TexComponentCount::ST ? TexMtx_Write_Float : TexMtx_Write_Float2);
      }
      else
      {
        WriteCall(TexMtx_Write_Float3);
      }
    }
    else
    {
      if (tc != VertexComponentFormat::NotPresent)
      {
        m_native_vtx_decl.texcoords[i].enable = true;
        m_native_vtx_decl.texcoords[i].components = elements == TexComponentCount::ST ? 2 : 1;
        nat_offset += 4 * (elements == TexComponentCount::ST ? 2 : 1);
      }
    }

    if (tc == VertexComponentFormat::NotPresent)
    {
      // if there's more tex coords later, have to write a dummy call
      bool has_more = false;
      for (size_t j = i + 1; j < m_VtxDesc.high.TexCoord.Size(); ++j)
      {
        if (m_VtxDesc.high.TexCoord[j] != VertexComponentFormat::NotPresent)
        {
          has_more = true;
          // Keep tcIndex in sync so that the correct array is used later
          WriteCall(VertexLoader_TextCoord::GetDummyFunction());
          break;
        }
        else if (m_VtxDesc.low.TexMatIdx[j])
        {
          has_more = true;
        }
      }
      if (!has_more)
      {
        // no more tex coords and tex matrices, so exit loop
        break;
      }
    }
  }

  // indexed position formats may skip the vertex
  if (IsIndexed(m_VtxDesc.low.Position))
  {
    WriteCall(SkipVertex);
  }

  m_native_vtx_decl.stride = nat_offset;
}

void VertexLoader::WriteCall(TPipelineFunction func)
{
  m_PipelineStages.push_back(func);
}

int VertexLoader::RunVertices(const u8* src, u8* dst, int count)
{
#if GALAXYPAD_INDEXED_VERTEX_BATCH
  static u64 total_vertices = 0, batch_vertices = 0, next_report = 1u << 20;
  total_vertices += count;
  const auto report = [&] {
    if (total_vertices >= next_report)
    {
      std::fprintf(stderr, "[indexed-vertex-batch] eligible=%llu total=%llu\n",
                   static_cast<unsigned long long>(batch_vertices),
                   static_cast<unsigned long long>(total_vertices));
      next_report = total_vertices + (1u << 20);
    }
  };
  // Match the complete compiled stage sequence; unsupported formats retain the
  // generic path. Resolve once per batch, never once per vertex.
  const auto* stage = m_PipelineStages.begin();
  const auto* end = m_PipelineStages.end();
  const bool matrix = stage != end && *stage == PosMtx_ReadDirect_UByte;
  if (matrix)
    ++stage;
  static const auto position = VertexLoader_Position::GetFunction(
      VertexComponentFormat::Index16, ComponentFormat::Short, CoordComponentCount::XYZ);
  static const auto normal = VertexLoader_Normal::GetFunction(
      VertexComponentFormat::Index16, ComponentFormat::Short, NormalComponentCount::N, false);
  if (end - stage >= 3 && stage[0] == position && stage[1] == normal)
  {
    stage += 2;
    static const auto color_function = VertexLoader_Color::GetFunction(
        VertexComponentFormat::Index16, ColorFormat::RGBA8888);
    const bool color = *stage == color_function;
    if (color)
      ++stage;
    static const auto texcoord = VertexLoader_TextCoord::GetFunction(
        VertexComponentFormat::Index16, ComponentFormat::Short, TexComponentCount::ST);
    int texcoords = 0;
    while (stage != end && *stage == texcoord && texcoords < 8)
    {
      ++stage;
      ++texcoords;
    }
    if (end - stage == 1 && *stage == SkipVertex)
    {
      batch_vertices += count;
      report();
      if (matrix)
        return color ? IndexedVertexBatch::Run<true, true>(this, src, dst, count, texcoords) :
                       IndexedVertexBatch::Run<true, false>(this, src, dst, count, texcoords);
      return color ? IndexedVertexBatch::Run<false, true>(this, src, dst, count, texcoords) :
                     IndexedVertexBatch::Run<false, false>(this, src, dst, count, texcoords);
    }
  }
  report();
#endif
  g_vertex_manager_write_ptr = dst;
  g_video_buffer_read_ptr = src;

  m_numLoadedVertices += count;
  m_skippedVertices = 0;

  for (m_remaining = count - 1; m_remaining >= 0; m_remaining--)
  {
    m_tcIndex = 0;
    m_colIndex = 0;
    m_texmtxwrite = m_texmtxread = 0;
    for (TPipelineFunction& func : m_PipelineStages)
      func(this);
    PRIM_LOG("\n");
  }

  return count - m_skippedVertices;
}
