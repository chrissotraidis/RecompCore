// Copyright 2013 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "Common/CommonTypes.h"
namespace SlippiWire { enum class MessageID : u8 {
  SLIPPI_PAD = 0x80,
  SLIPPI_PAD_ACK = 0x81,
  SLIPPI_MATCH_SELECTIONS = 0x82,
  SLIPPI_CONN_SELECTED = 0x83,
  SLIPPI_CHAT_MESSAGE = 0x84,
  SLIPPI_COMPLETE_STEP = 0x85,
  SLIPPI_SYNCED_STATE = 0x86,
}; }
