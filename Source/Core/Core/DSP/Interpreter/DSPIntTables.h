// Copyright 2018 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>

#include "Core/DSP/DSPCommon.h"

namespace DSP::Interpreter
{
class Interpreter;

using InterpreterFunction = void (Interpreter::*)(UDSPInstruction);

struct DecodedInterpreterOp
{
  InterpreterFunction main;
  InterpreterFunction extension;
  bool extended;
};

InterpreterFunction GetOp(UDSPInstruction inst);
InterpreterFunction GetExtOp(UDSPInstruction inst);

// One entry per 16-bit opcode: the main handler, its extension handler and the
// extended flag, predecoded so that dispatch is an index rather than a search.
// Built by InitInstructionTables, which the interpreter's constructor calls
// once, from the two op tables that function fills - so the decoded table is a
// pure function of them and needs no guard of its own. It used to be a
// function-local static in GetDecodedOp, which put a thread-safe-initialization
// guard (adrp, add, ldaprb, tbz) and a call in front of every dispatched DSP
// instruction; the built host is in docs/status/CURRENT.md, 2026-09-22.
extern std::array<DecodedInterpreterOp, 65536> s_decoded_ops;

inline const DecodedInterpreterOp& GetDecodedOp(UDSPInstruction inst)
{
  return s_decoded_ops[inst];
}

void InitInstructionTables();
}  // namespace DSP::Interpreter
