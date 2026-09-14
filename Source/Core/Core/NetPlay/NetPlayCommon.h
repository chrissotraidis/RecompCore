// Copyright 2021 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <SFML/Network/Packet.hpp>

#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "Common/CommonTypes.h"

namespace NetPlay
{
using namespace std::chrono_literals;
constexpr u32 CANONICAL_RAM_REGION_SIZE = 1024 * 1024;
constexpr size_t CANONICAL_RAM_REGION_COUNT = 32;
// An arbitrary amount of time of no acknowledgement of sent packets before netplay decides a
// connection is disconnected
constexpr std::chrono::milliseconds PEER_TIMEOUT = 30s;

// Architectural state captured at the configured guest idle PC. Unlike the
// ordinary TimeBase packet fields, every value here comes from one explicit
// emulated boundary and can therefore be compared across host architectures.
struct CanonicalStateSnapshot
{
  u64 sequence = 0;
  u32 guest_pc = 0;
  u64 timebase = 0;
  u64 state_hash = 0;
  u64 integer_state_hash = 0;
  u64 fpr_state_hash = 0;
  u64 paired_state_hash = 0;
  u64 ram_hash = 0;
  std::array<u64, CANONICAL_RAM_REGION_COUNT> ram_region_hashes{};
};

inline bool CanonicalStateMatches(const CanonicalStateSnapshot& lhs,
                                  const CanonicalStateSnapshot& rhs)
{
  return lhs.sequence != 0 && lhs.sequence == rhs.sequence && lhs.guest_pc == rhs.guest_pc &&
         lhs.timebase == rhs.timebase && lhs.state_hash == rhs.state_hash &&
         lhs.integer_state_hash == rhs.integer_state_hash &&
         lhs.fpr_state_hash == rhs.fpr_state_hash &&
         lhs.paired_state_hash == rhs.paired_state_hash && lhs.ram_hash == rhs.ram_hash &&
         lhs.ram_region_hashes == rhs.ram_region_hashes;
}

inline size_t CanonicalRamFirstDifferingRegion(const CanonicalStateSnapshot& lhs,
                                               const CanonicalStateSnapshot& rhs)
{
  for (size_t i = 0; i < CANONICAL_RAM_REGION_COUNT; ++i)
  {
    if (lhs.ram_region_hashes[i] != rhs.ram_region_hashes[i])
      return i;
  }
  return CANONICAL_RAM_REGION_COUNT;
}

bool CompressFileIntoPacket(const std::string& file_path, sf::Packet& packet);
bool CompressFolderIntoPacket(const std::string& folder_path, sf::Packet& packet);
bool CompressBufferIntoPacket(std::span<const u8> in_buffer, sf::Packet& packet);
bool DecompressPacketIntoFile(sf::Packet& packet, const std::string& file_path);
bool DecompressPacketIntoFolder(sf::Packet& packet, const std::string& folder_path);
std::optional<std::vector<u8>> DecompressPacketIntoBuffer(sf::Packet& packet);
}  // namespace NetPlay
