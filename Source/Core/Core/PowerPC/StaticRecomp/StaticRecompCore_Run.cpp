// RecompCore: StaticRecomp CPU core - Main execution loop.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/StaticRecomp/StaticRecompCore.h"
#include "Core/System.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/PowerPC/Interpreter/Interpreter.h"
#include "Core/PowerPC/StaticRecomp/StaticRecompLockstep.h"
#include "Core/CoreTiming.h"
#include "Core/HW/CPU.h"
#include "Core/Config/MainSettings.h"
#include "Core/Config/ConfigManager.h"
#include "Core/HW/SystemTimers.h"
#include "Core/PowerPC/StaticRecomp/RunCost.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace
{
constexpr u32 SYNC_EXCEPTION_MASK = ~static_cast<u32>(
    EXCEPTION_EXTERNAL_INT | EXCEPTION_DECREMENTER | EXCEPTION_PERFORMANCE_MONITOR);
constexpr u32 ASYNC_EXCEPTION_MASK =
    EXCEPTION_EXTERNAL_INT | EXCEPTION_DECREMENTER | EXCEPTION_PERFORMANCE_MONITOR;
constexpr u32 MSR_EE = 0x00008000u;

struct FileCloser
{
  void operator()(std::FILE* file) const
  {
    if (file)
      std::fclose(file);
  }
};

using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

FilePtr OpenDispatchTrace()
{
  const char* path = std::getenv("STATICRECOMP_TRACE_FILE");
  if (!path || !*path)
    return {};

  FilePtr file(std::fopen(path, "w"));
  if (file)
  {
    std::fprintf(file.get(), "dispatch,pc,lr,ctr,cr,timebase,ppc_downcount\n");
    std::fflush(file.get());
  }
  return file;
}
}

int StaticRecompCore::HookDirectCallBoundary(CPUState* cpu, u32 address)
{
  if (!cpu || !cpu->external_user_data)
    return 0;
  auto* core = static_cast<StaticRecompCore*>(cpu->external_user_data);
  if (cpu != &core->m_guest || !core->m_direct_boundary_enabled ||
      !core->m_in_native_dispatch || core->m_direct_must_yield)
    return 0;

  auto& ppc = core->m_system.GetPowerPC().GetPPCState();
  const s64 charge = -cpu->downcount;
  cpu->downcount = 0;
  const u64 effective_charge = charge > 0 ? static_cast<u64>(charge) :
                                             (core->m_direct_segment_committed ? 0 : 1);
  ppc.downcount -= static_cast<int>(effective_charge);
  core->m_charged_cycles += effective_charge;
  core->AdvanceGuestTimebase(effective_charge);
  core->m_direct_segment_committed = true;
  ++core->m_direct_boundary_checks;

  // A failed check returns through all native callers. Do not let a matching
  // outer continuation accidentally resume after an inner transfer was denied.
  core->m_direct_must_yield = true;
  if (!core->m_module_active || core->m_has_rel_modules ||
      core->m_lockstep_verifier->IsEnabled() || cpu->pc != address ||
      *core->m_system.GetCPU().GetStatePtr() != CPU::State::Running ||
      ppc.downcount <= 0 || cpu->exception ||
      (ppc.Exceptions & SYNC_EXCEPTION_MASK) != 0 ||
      ((ppc.Exceptions & ASYNC_EXCEPTION_MASK) != 0 && (cpu->msr & MSR_EE) != 0) ||
      (core->m_idle_pc != 0 && address == core->m_idle_pc) ||
      !core->FastDispatchableAt(address) || core->IsHostCallAddress(address))
    return 0;

  // The target (callee or caller continuation) starts a new charged segment.
  core->m_direct_must_yield = false;
  core->m_direct_segment_committed = false;
  ++core->m_direct_transfers;
  return 1;
}

void StaticRecompCore::Run()
{
  auto& core_timing = m_system.GetCoreTiming();
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();
  auto& interpreter = m_system.GetInterpreter();
  auto& memory = m_system.GetMemory();
  const CPU::State* state_ptr = m_system.GetCPU().GetStatePtr();
  FilePtr dispatch_trace = OpenDispatchTrace();

  m_guest.ram = memory.GetRAM();
  m_guest.ram_size = memory.GetRamSizeReal();
  m_guest.exram = memory.GetEXRAM();
  m_guest.exram_size = memory.GetExRamSizeReal();
  InitLookupTable(m_guest.ram_size, m_guest.exram_size);
  const bool lockstep_enabled = m_lockstep_verifier->IsEnabled();
  const auto fast_dispatchable_at = [this](u32 address) {
    if (m_has_rel_modules || !m_forced_fallback_ranges.empty())
      return FastDispatchableAt(address);
    if (!m_module_active || m_chunk_lookup_table.empty())
      return false;

    int lookup_index = -1;
    if (address >= 0x80000000u && address < 0x80000000u + m_lookup_ram_size)
    {
      lookup_index = static_cast<int>((address - 0x80000000u) >> 2);
    }
    else if (address >= 0x90000000u && address < 0x90000000u + m_lookup_exram_size)
    {
      lookup_index = static_cast<int>((m_lookup_ram_size >> 2) + ((address - 0x90000000u) >> 2));
    }
    if (lookup_index < 0 || lookup_index >= static_cast<int>(m_chunk_lookup_table.size()))
      return false;
    const int chunk = m_chunk_lookup_table[lookup_index];
    return chunk >= 0 && m_chunk_state[chunk] == CHUNK_VERIFIED;
  };

  const std::string initial_game_id = SConfig::GetInstance().GetGameID();
  m_module_active = m_module && (initial_game_id.empty() || initial_game_id == m_module->game_id);

  if (!m_module_active && m_fallback_jit && !m_guest.host_call)
  {
    m_fallback_jit->Run();
    return;
  }

  // Opt-in phase-selected CPU-time attribution, not per-instruction timing.
  std::unique_ptr<galaxypad::diagnostics::RunCost> run_cost;
  const char* cost_enabled = std::getenv("GALAXYPAD_RUN_COST");
  const char* cost_marker = std::getenv("GALAXYPAD_RUN_COST_START_FILE");
  std::string cost_start_file = cost_enabled && std::strcmp(cost_enabled, "1") == 0 &&
      cost_marker && *cost_marker ? cost_marker : "";
  u32 cost_poll = 0;
  while (*state_ptr == CPU::State::Running)
  {
    core_timing.Advance();
    if (!cost_start_file.empty() && (++cost_poll & 1023u) == 0)
    {
      if (std::FILE* marker = std::fopen(cost_start_file.c_str(), "rb"))
      {
        std::fclose(marker);
        run_cost = std::make_unique<galaxypad::diagnostics::RunCost>();
        cost_start_file.clear();
        std::fprintf(stderr, "[galaxypad-run-cost] capture-start pc=%08x\n", ppc.pc);
      }
    }
    // Diagnostic only: poll at timing-slice boundaries, never per guest step.
    // Caller supplies a fresh absent path, then creates it after visual scene
    // verification. Do not delete caller files or include startup in the census.
    if (!m_fallback_start_file.empty() && (++m_fallback_start_poll & 1023u) == 0)
    {
      if (std::FILE* marker = std::fopen(m_fallback_start_file.c_str(), "rb"))
      {
        std::fclose(marker);
        m_fallback_histogram = std::make_unique<galaxypad::diagnostics::FallbackHistogram<>>();
        m_fallback_start_file.clear();
        galaxypad::diagnostics::DumpExceptionVectors(stderr, memory.GetRAM(), memory.GetRamSizeReal());
        std::fprintf(stderr, "[galaxypad-fallback-pcs] capture-start pc=%08x native=%llu\n",
                     ppc.pc, static_cast<unsigned long long>(m_native_dispatches));
      }
    }
    const std::string current_game_id = SConfig::GetInstance().GetGameID();
    m_module_active = m_module && (current_game_id.empty() || current_game_id == m_module->game_id);

    do
    {
      // MSR.FP needs no gate here: generated FPU instructions raise the
      // FP-unavailable exception themselves (ppc_fp_available).
      if (m_module_active && DispatchableAt(ppc.pc) &&
          !(m_guest.host_call && IsHostCallAddress(ppc.pc)))
      {
        galaxypad::diagnostics::RunCost::Scope cost_span(
            run_cost.get(), galaxypad::diagnostics::RunLane::Native);
        SyncIn();
        ++m_bursts;
        do
        {
          if (dispatch_trace && (m_native_dispatches & 0xFFFFFu) == 0)
          {
            std::fprintf(dispatch_trace.get(), "%llu,%08x,%08x,%08x,%08x,%llu,%d\n",
                         static_cast<unsigned long long>(m_native_dispatches), m_guest.pc,
                         m_guest.lr, m_guest.ctr, m_guest.cr,
                         static_cast<unsigned long long>(m_guest.timebase), ppc.downcount);
            std::fflush(dispatch_trace.get());
          }
          const bool do_ls = lockstep_enabled && m_lockstep_verifier->ShouldCheck(m_guest.pc);
          if (do_ls)
          {
            m_lockstep_verifier->Prepare(m_guest);
          }

          if (m_collect_dispatch_samples && (m_native_dispatches & 4095u) == 0)
            ++m_dispatch_samples[m_guest.pc];
          const u32 runtime_dispatch_address = m_guest.pc;
          u32 linked_dispatch_address = runtime_dispatch_address;
          if (m_has_rel_modules)
            ResolveNativeAddress(runtime_dispatch_address, &linked_dispatch_address, nullptr);
          m_guest.pc = linked_dispatch_address;
          if (m_direct_boundary_enabled)
          {
            m_direct_segment_committed = false;
            m_direct_must_yield = false;
            m_in_native_dispatch = true;
          }
          m_module->dispatch(&m_guest, linked_dispatch_address);
          m_in_native_dispatch = false;
          if (m_has_rel_modules)
            m_guest.pc = TranslateRelAddress(m_guest.pc);
          ++m_native_dispatches;

          if (do_ls)
          {
            m_lockstep_verifier->Verify(m_guest);
          }

          // Flush the module's per-block cycle charges into Dolphin's
          // downcount. A dispatch that charged nothing (PC-switch default,
          // pure embedded data) still costs 1 so the burst always makes
          // downcount progress; this per-dispatch flush is also the
          // dispatcher back-edge timing check — CoreTiming regains control
          // with at least CachedInterpreter's per-block frequency, so
          // external-interrupt latency matches stock.
          const s64 charge = -m_guest.downcount;
          m_guest.downcount = 0;
          // A rejected direct boundary has already charged its last segment.
          // Its glue-only unwind must not add the ordinary empty-dispatch cycle.
          const u64 effective_charge = charge > 0 ? static_cast<u64>(charge) :
                                                     (m_direct_segment_committed ? 0 : 1);
          ppc.downcount -= static_cast<int>(effective_charge);
          m_charged_cycles += effective_charge;
          const u64 total_cycles = m_timebase_cycle_remainder + effective_charge;
          m_guest.timebase += total_cycles / SystemTimers::TIMER_RATIO;
          m_timebase_cycle_remainder = total_cycles % SystemTimers::TIMER_RATIO;

          // Idle loop skipping for configured target loops (e.g. Wii Menu OSIdleThread)
          if (m_guest.pc == m_idle_pc && m_idle_pc != 0)
          {
            m_system.GetCoreTiming().Idle();
          }

          // ctx->timebase is refreshed at burst start (SyncIn), and here we
          // incrementally advance it by the exact block cycle charges to
          // prevent guest busy-wait loops from spinning on a stale timebase.
          if (m_guest.exception)
          {
            // DolRecomp's runtime already redirected pc/msr/srr to the guest
            // exception vector; the flag only signals that it happened.
            m_guest.exception = 0;
            m_guest.program_exception = 0;
            ++m_native_exceptions;
          }
          if ((ppc.Exceptions & SYNC_EXCEPTION_MASK) != 0)
            break;  // Hook-raised synchronous exception: deliver via Dolphin below.
          if ((ppc.Exceptions & ASYNC_EXCEPTION_MASK) != 0 && (m_guest.msr & MSR_EE) != 0)
            break;  // rfi/mtmsr re-enabled interrupts while one was pending.
        } while (m_module_active && fast_dispatchable_at(m_guest.pc) &&
                 !(m_guest.host_call && IsHostCallAddress(m_guest.pc)) && ppc.downcount > 0 &&
                 *state_ptr == CPU::State::Running);
        SyncOut();
        if ((ppc.Exceptions & SYNC_EXCEPTION_MASK) != 0)
          power_pc.CheckExceptions();
        else if ((ppc.Exceptions & ASYNC_EXCEPTION_MASK) != 0)
          power_pc.CheckExternalExceptions();
      }
      else
      {
        // Includes host-call routing, forced fallback and interpreter/JIT work.
        // It is deliberately NOT labeled pure interpreter or vector execution.
        galaxypad::diagnostics::RunCost::Scope cost_span(
            run_cost.get(), galaxypad::diagnostics::RunLane::NonNativeRouting);
        if (m_guest.host_call && IsHostCallAddress(ppc.pc))
        {
          SyncIn();
          bool handled = m_guest.host_call(&m_guest, m_guest.pc);
          if (!handled && m_guest.pc < m_guest.ram_size)
            handled = m_guest.host_call(&m_guest, m_guest.pc | 0x80000000u);
          if (m_fallback_jit && IsHostCallAddress(m_guest.lr))
            m_fallback_jit->GetBlockCache()->InvalidateICache(m_guest.lr, 4, true);
          if (handled)
          {
            const s64 charge = -m_guest.downcount;
            m_guest.downcount = 0;
            const u64 effective_charge = static_cast<u64>(charge > 0 ? charge : 1);
            ppc.downcount -= static_cast<int>(effective_charge);
            AdvanceGuestTimebase(effective_charge);
            SyncOut();
            continue;
          }
          SyncOut();
          if (m_fallback_jit)
          {
            m_host_call_passthrough_pc = ppc.pc;
            m_host_call_passthrough = true;
          }
        }
        // SingleStepInner delivers synchronous exceptions itself; external
        // interrupts are delivered at slice start, as in Interpreter::Run.
        if (m_module_active && IsForcedFallbackAddress(ppc.pc))
        {
          if (m_fallback_histogram)
            m_fallback_histogram->Record(ppc.pc, galaxypad::diagnostics::FallbackPath::Forced);
          ppc.downcount -= interpreter.SingleStepInner();
          ++m_fallback_steps;
        }
        else if (m_fallback_jit)
        {
          m_fallback_jit->Run();
        }
        else
        {
          do
          {
            if (m_fallback_histogram)
              m_fallback_histogram->Record(ppc.pc, galaxypad::diagnostics::FallbackPath::Uncovered);
            ppc.downcount -= interpreter.SingleStepInner();
            ++m_fallback_steps;
          } while (!(m_module_active && DispatchableAt(ppc.pc)) &&
                   !IsHostCallAddress(ppc.pc) && ppc.downcount > 0 &&
                   *state_ptr == CPU::State::Running);
        }
      }
    } while (ppc.downcount > 0 && *state_ptr == CPU::State::Running);
  }
  if (run_cost)
    run_cost->Report(stderr); // Still on CPU thread; all branch scopes have ended.
}

void StaticRecompCore::SingleStep()
{
  // Debugger stepping runs through the interpreter; state outside Run() lives
  // in PowerPCState, so no sync is needed.
  auto& system = m_system;
  system.GetCoreTiming().Advance();
  system.GetPPCState().downcount -= system.GetInterpreter().SingleStepInner();
}
