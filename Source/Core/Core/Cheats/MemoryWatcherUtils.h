// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "Common/CommonTypes.h"

namespace MemoryWatcherUtils
{
namespace detail
{
struct MemoryLocation
{
  std::span<const u8> memory;
  u32 offset;
};

inline std::optional<MemoryLocation> ResolveStaticRecompAddress(std::span<const u8> mem1,
                                                               std::span<const u8> mem2,
                                                               u32 address)
{
  const u32 segment = address & 0xf0000000u;
  if (segment == 0x80000000u || segment == 0xc0000000u)
  {
    const u32 offset = address & 0x0fffffffu;
    if (offset < mem1.size())
      return MemoryLocation{mem1, offset};
  }
  else if (segment == 0x90000000u || segment == 0xd0000000u)
  {
    const u32 offset = address & 0x0fffffffu;
    if (offset < mem2.size())
      return MemoryLocation{mem2, offset};
  }
  return std::nullopt;
}
}  // namespace detail

inline bool IsStaticRecompRAMAddress(std::span<const u8> mem1, std::span<const u8> mem2,
                                     u32 address)
{
  return detail::ResolveStaticRecompAddress(mem1, mem2, address).has_value();
}

inline std::optional<u32> ReadStaticRecompU32(std::span<const u8> mem1,
                                             std::span<const u8> mem2, u32 address)
{
  const auto location = detail::ResolveStaticRecompAddress(mem1, mem2, address);
  if (!location || location->memory.size() - location->offset < sizeof(u32))
    return std::nullopt;

  const u8* const bytes = location->memory.data() + location->offset;
  return (static_cast<u32>(bytes[0]) << 24) | (static_cast<u32>(bytes[1]) << 16) |
         (static_cast<u32>(bytes[2]) << 8) | static_cast<u32>(bytes[3]);
}

inline bool ShouldPublish(const std::optional<u32>& current_value, u32 new_value)
{
  return !current_value || *current_value != new_value;
}
}  // namespace MemoryWatcherUtils
