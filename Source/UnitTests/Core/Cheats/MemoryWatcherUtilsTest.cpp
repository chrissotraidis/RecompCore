// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <optional>

#include <gtest/gtest.h>

#include "Core/Cheats/MemoryWatcherUtils.h"

namespace
{
using MemoryWatcherUtils::ReadStaticRecompU32;
using MemoryWatcherUtils::ShouldPublish;

TEST(MemoryWatcherUtils, ReadsBigEndianCachedAndUncachedMem1)
{
  const std::array<u8, 8> mem1{{0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x56, 0x78}};
  const std::array<u8, 1> mem2{};

  EXPECT_EQ(ReadStaticRecompU32(mem1, mem2, 0x80000004), 0x12345678u);
  EXPECT_EQ(ReadStaticRecompU32(mem1, mem2, 0xc0000004), 0x12345678u);
}

TEST(MemoryWatcherUtils, ReadsBigEndianCachedAndUncachedMem2)
{
  const std::array<u8, 1> mem1{};
  const std::array<u8, 8> mem2{{0x00, 0x00, 0x00, 0x00, 0x89, 0xab, 0xcd, 0xef}};

  EXPECT_EQ(ReadStaticRecompU32(mem1, mem2, 0x90000004), 0x89abcdefu);
  EXPECT_EQ(ReadStaticRecompU32(mem1, mem2, 0xd0000004), 0x89abcdefu);
}

TEST(MemoryWatcherUtils, RejectsInvalidAndTruncatedRanges)
{
  const std::array<u8, 8> mem1{};
  const std::array<u8, 8> mem2{};

  EXPECT_EQ(ReadStaticRecompU32(mem1, mem2, 0x7ffffffcu), std::nullopt);
  EXPECT_EQ(ReadStaticRecompU32(mem1, mem2, 0x80000005u), std::nullopt);
  EXPECT_EQ(ReadStaticRecompU32(mem1, mem2, 0xffffffffu), std::nullopt);
}

TEST(MemoryWatcherUtils, PublishesInitialZeroThenSuppressesUnchangedValue)
{
  EXPECT_TRUE(ShouldPublish(std::nullopt, 0));
  EXPECT_FALSE(ShouldPublish(std::optional<u32>{0}, 0));
  EXPECT_TRUE(ShouldPublish(std::optional<u32>{0}, 1));
}
}  // namespace
