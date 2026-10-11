// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "Common/CommonTypes.h"
#include "Common/FileUtil.h"
#include "Common/Swap.h"
#include "Core/HW/EXI/EXI.h"
#include "Core/HW/EXI/EXI_Device.h"
#include "Core/HW/GCMemcard/GCMemcard.h"
#include "Core/HW/GCMemcard/GCMemcardDirectory.h"

namespace
{
constexpr u16 SAVE_BLOCKS = 4;
constexpr u8 OLD_FILL = 0x11;
constexpr u8 NEW_FILL = 0x22;
constexpr u8 NO_SAVE = 0xFF;
constexpr char FILE_NAME[] = "01-GMSE-super_mario_sunshine.gci";

Memcard::DEntry MakeEntry()
{
  Memcard::DEntry entry;
  std::memcpy(entry.m_gamecode.data(), "GMSE", 4);
  std::memcpy(entry.m_makercode.data(), "01", 2);
  entry.m_filename.fill(0);
  std::memcpy(entry.m_filename.data(), "super_mario_sunshine", 20);
  entry.m_block_count = SAVE_BLOCKS;
  return entry;
}

std::vector<Memcard::GCMBlock> MakeBlocks(u8 fill)
{
  std::vector<Memcard::GCMBlock> blocks(SAVE_BLOCKS);
  for (Memcard::GCMBlock& block : blocks)
    block.m_block.fill(fill);
  return blocks;
}

// Loads the folder the way the emulator does at boot and returns the fill byte of the GMSE01
// save it found, NO_SAVE when none loaded, or 0 when the blocks disagree.
u8 LoadedFill(const std::string& directory)
{
  Memcard::HeaderData header_data;
  Memcard::InitializeHeaderData(&header_data, {}, Memcard::MBIT_SIZE_MEMORY_CARD_2043, false, 0,
                                0, 0);
  GCMemcardDirectory card(directory, ExpansionInterface::Slot::A, header_data,
                          Common::swap32(reinterpret_cast<const u8*>("GMSE")));
  std::array<u8, Memcard::BLOCK_SIZE> data;
  u8 fill = 0;
  for (u32 i = 0; i < SAVE_BLOCKS; ++i)
  {
    card.Read((Memcard::MC_FST_BLOCKS + i) * Memcard::BLOCK_SIZE, Memcard::BLOCK_SIZE,
              data.data());
    if (i == 0)
      fill = data[0];
    for (u8 byte : data)
    {
      if (byte != fill)
        return 0;
    }
  }
  return fill;
}

class GCMemcardDirectoryTest : public testing::Test
{
protected:
  void SetUp() override { m_directory = File::CreateTempDir() + "/"; }
  void TearDown() override { File::DeleteDirRecursively(m_directory); }

  std::string Path() const { return m_directory + FILE_NAME; }

  std::string m_directory;
};
}  // namespace

TEST_F(GCMemcardDirectoryTest, FullWriteLoads)
{
  ASSERT_EQ(Memcard::WriteGCIFile(Path(), MakeEntry(), MakeBlocks(OLD_FILL)),
            Memcard::GCIWriteResult::Success);
  EXPECT_EQ(LoadedFill(m_directory), OLD_FILL);
}

// A save cut short at any point must leave either the old save or the new one.
TEST_F(GCMemcardDirectoryTest, StopDuringWriteKeepsOldOrNewSave)
{
  for (size_t stop = 0; stop < SAVE_BLOCKS; ++stop)
  {
    ASSERT_EQ(Memcard::WriteGCIFile(Path(), MakeEntry(), MakeBlocks(OLD_FILL)),
              Memcard::GCIWriteResult::Success);
    EXPECT_EQ(Memcard::WriteGCIFile(Path(), MakeEntry(), MakeBlocks(NEW_FILL), stop),
              Memcard::GCIWriteResult::WriteFailed);
    const u8 fill = LoadedFill(m_directory);
    EXPECT_TRUE(fill == OLD_FILL || fill == NEW_FILL)
        << "write stopped after " << stop << " blocks left fill " << static_cast<int>(fill);
  }
}
