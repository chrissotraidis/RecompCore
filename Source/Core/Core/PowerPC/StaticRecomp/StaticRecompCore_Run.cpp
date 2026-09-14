// RecompCore: StaticRecomp CPU core - Main execution loop.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/StaticRecomp/StaticRecompCore.h"
#include "Core/System.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/PowerPC/Interpreter/Interpreter.h"
#include "Core/PowerPC/StaticRecomp/StaticRecompLockstep.h"
#include "Core/CoreTiming.h"
#include "Core/HW/CPU.h"
#include "Core/NetPlay/NetPlayProto.h"
#include "Core/Config/MainSettings.h"
#include "Core/Config/ConfigManager.h"
#include "Core/HW/SystemTimers.h"
#include "Common/FramePhaseTiming.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace
{
constexpr u32 SYNC_EXCEPTION_MASK = ~static_cast<u32>(
    EXCEPTION_EXTERNAL_INT | EXCEPTION_DECREMENTER | EXCEPTION_PERFORMANCE_MONITOR);

u64 GetCurrentThreadCpuNanoseconds()
{
  timespec time{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time);
  return static_cast<u64>(time.tv_sec) * 1'000'000'000 + static_cast<u64>(time.tv_nsec);
}

}  // namespace

void StaticRecompCore::UpdateProfileCapture()
{
  if (m_profile_trigger_address == 0 || m_profile_capture_complete)
    return;

  const u32 value = m_system.GetMemory().Read_U32(m_profile_trigger_address);
  const bool matches = (value & m_profile_trigger_mask) == m_profile_trigger_value;
  if (matches && !m_profile_capture_active)
  {
    m_profile_reset();
    m_profile_capture_active = true;
    std::fprintf(stderr, "[staticrecomp] profile capture started\n");
  }
  else if (!matches && m_profile_capture_active)
  {
    const int result = m_profile_dump();
    m_profile_capture_active = false;
    m_profile_capture_complete = true;
    std::fprintf(stderr, "[staticrecomp] profile capture dumped: result=%d\n", result);
  }
}

void StaticRecompCore::Run()
{
  const bool diagnostics = Common::FramePhaseTiming::IsEnabled() ||
      m_lockstep_verifier->IsEnabled() || !m_dispatch_frame_log_path.empty() ||
      !m_dispatch_time_log_path.empty() || !m_dispatch_burst_log_path.empty() ||
      std::getenv("STATICRECOMP_DISPATCH_SAMPLE") != nullptr ||
      std::getenv("STATICRECOMP_FREEZE_TRACE") != nullptr;
  if (diagnostics)
    RunWithDiagnostics<true>();
  else
    RunWithDiagnostics<false>();
}

template <bool Diagnostics>
void StaticRecompCore::RunWithDiagnostics()
{
  auto& core_timing = m_system.GetCoreTiming();
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();
  auto& interpreter = m_system.GetInterpreter();
  auto& memory = m_system.GetMemory();
  const CPU::State* state_ptr = m_system.GetCPU().GetStatePtr();
  const bool log_phase = Diagnostics && Common::FramePhaseTiming::IsEnabled();
  const bool lockstep_enabled = Diagnostics && m_lockstep_verifier->IsEnabled();
  const bool sample_dispatches = Diagnostics && (!m_dispatch_frame_log_path.empty() ||
                                 !m_dispatch_time_log_path.empty() ||
                                 std::getenv("STATICRECOMP_DISPATCH_SAMPLE") != nullptr);
  const bool sample_dispatch_time = Diagnostics && !m_dispatch_time_log_path.empty();
  const bool sample_dispatch_bursts = Diagnostics && !m_dispatch_burst_log_path.empty();
  const bool freeze_trace = Diagnostics && std::getenv("STATICRECOMP_FREEZE_TRACE") != nullptr;
  const bool has_rel_modules = m_module && m_module->num_rel_modules != 0;
  const u32 idle_pc = m_idle_pc;
  const u32 secondary_idle_pc = m_secondary_idle_pc;

  m_guest.ram = memory.GetRAM();
  m_guest.ram_size = memory.GetRamSizeReal();
  m_guest.exram = memory.GetEXRAM();
  m_guest.exram_size = memory.GetExRamSizeReal();
  InitLookupTable(m_guest.ram_size, m_guest.exram_size);

  const std::string initial_game_id = SConfig::GetInstance().GetGameID();
  m_module_active = m_module && (initial_game_id.empty() || initial_game_id == m_module->game_id);

  if (!m_module_active && m_fallback_jit && !m_guest.host_call)
  {
    m_fallback_jit->Run();
    return;
  }

  while (*state_ptr == CPU::State::Running)
  {
    const TimePoint cpu_slice_start = log_phase ? Clock::now() : TimePoint{};
    const u64 cpu_thread_start = log_phase ? GetCurrentThreadCpuNanoseconds() : 0;
    const u64 phase_bursts = m_bursts;
    const u64 phase_cycles = m_charged_cycles;
    const u64 phase_native_dispatches = m_native_dispatches;
    const u64 phase_fallback_steps = m_fallback_steps;
    const u64 phase_hook_fallbacks = m_hook_fallback_instructions;
    core_timing.Advance();
    UpdateProfileCapture();
    const std::string current_game_id = SConfig::GetInstance().GetGameID();
    m_module_active = m_module && (current_game_id.empty() || current_game_id == m_module->game_id);

    do
    {
      // MSR.FP needs no gate here: generated FPU instructions raise the
      // FP-unavailable exception themselves (ppc_fp_available).
      if (m_module_active && DispatchableAt(ppc.pc) &&
          !(m_guest.host_call && IsHostCallAddress(ppc.pc)))
      {
        SyncIn();
        ++m_bursts;
        do
        {
          if (freeze_trace &&
              (m_guest.pc == 0x80191ee8u || m_guest.pc == 0x80191f00u ||
               m_guest.pc == 0x80191858u))
            std::fprintf(stderr, "[freeze-trace] pc=%08x r3=%08x r4=%08x lr=%08x\n",
                         m_guest.pc, m_guest.gpr[3], m_guest.gpr[4], m_guest.lr);
          const bool do_ls = lockstep_enabled && m_lockstep_verifier->ShouldCheck(m_guest.pc);
          if (do_ls)
          {
            m_lockstep_verifier->Prepare(m_guest);
          }

          const bool selected_dispatch =
              sample_dispatches &&
              m_native_dispatches % m_dispatch_sample_interval == m_dispatch_sample_offset;
          const u32 sampled_dispatch_pc = m_guest.pc;
          if (selected_dispatch)
          {
            ++m_dispatch_samples[m_guest.pc];
            if (!m_dispatch_frame_log_path.empty())
            {
              m_dispatch_frame_samples.push_back(
                  {Common::FramePhaseTiming::GetPresentFrameIndex(), m_guest.pc});
              if (m_dispatch_frame_samples.size() >= 16384)
                FlushDispatchFrameSamples(false);
            }
          }
          if (sample_dispatch_bursts)
          {
            // Retain 16 consecutive entries every 16,384 dispatches. This is
            // sparse enough for attribution but preserves real path edges.
            if ((m_native_dispatches & 16383u) == 0)
            {
              ++m_dispatch_burst_id;
              m_dispatch_burst_index = 0;
              m_dispatch_burst_remaining = 16;
            }
            if (m_dispatch_burst_remaining != 0)
            {
              m_dispatch_burst_samples.push_back(
                  {Common::FramePhaseTiming::GetPresentFrameIndex(),
                   Common::FramePhaseTiming::GetEmulatedFrameIndex(), m_dispatch_burst_id,
                   m_dispatch_burst_index++, m_dispatch_burst_previous_pc, m_guest.pc});
              --m_dispatch_burst_remaining;
              if (m_dispatch_burst_samples.size() >= 16384)
                FlushDispatchBurstSamples(false);
            }
            m_dispatch_burst_previous_pc = m_guest.pc;
          }
          const u32 runtime_dispatch_address = m_guest.pc;
          u32 linked_dispatch_address = runtime_dispatch_address;
          if (has_rel_modules)
            ResolveNativeAddress(runtime_dispatch_address, &linked_dispatch_address, nullptr);
          m_guest.pc = linked_dispatch_address;
          const TimePoint dispatch_time_start =
              selected_dispatch && sample_dispatch_time ? Clock::now() : TimePoint{};
          if (std::getenv("STATICRECOMP_REGISTER_TRACE") &&
              (runtime_dispatch_address == 0x8001990cu ||
               runtime_dispatch_address == 0x80019940u ||
               runtime_dispatch_address == 0x80019980u ||
               runtime_dispatch_address == 0x80343680u ||
               runtime_dispatch_address == 0x8065cc80u ||
               runtime_dispatch_address == 0x8065cce0u))
          {
            std::fprintf(stderr,
                         "[staticrecomp] register-trace before pc=%08x linked=%08x "
                         "r0=%08x r1=%08x r3=%08x r4=%08x r5=%08x r7=%08x r8=%08x "
                         "r9=%08x lr=%08x cr=%08x\n",
                         runtime_dispatch_address, linked_dispatch_address, m_guest.gpr[0],
                         m_guest.gpr[1], m_guest.gpr[3], m_guest.gpr[4],
                         m_guest.gpr[5], m_guest.gpr[7], m_guest.gpr[8], m_guest.gpr[9],
                         m_guest.lr, m_guest.cr);
          }
          m_module->dispatch(&m_guest, linked_dispatch_address);
          if (std::getenv("STATICRECOMP_REGISTER_TRACE") && m_guest.pc == 0)
          {
            std::fprintf(stderr,
                         "[staticrecomp] register-trace dispatch-zero from=%08x linked=%08x "
                         "r0=%08x r1=%08x r3=%08x r4=%08x r5=%08x r7=%08x r8=%08x "
                         "r9=%08x lr=%08x cr=%08x srr0=%08x srr1=%08x msr=%08x\n",
                         runtime_dispatch_address, linked_dispatch_address, m_guest.gpr[0],
                         m_guest.gpr[1], m_guest.gpr[3], m_guest.gpr[4], m_guest.gpr[5],
                         m_guest.gpr[7], m_guest.gpr[8], m_guest.gpr[9], m_guest.lr, m_guest.cr,
                         m_guest.srr0, m_guest.srr1, m_guest.msr);
          }
          if (std::getenv("STATICRECOMP_REGISTER_TRACE") &&
              (runtime_dispatch_address == 0x8001990cu ||
               runtime_dispatch_address == 0x80019940u ||
               runtime_dispatch_address == 0x80019980u ||
               runtime_dispatch_address == 0x80343680u ||
               runtime_dispatch_address == 0x8065cc80u ||
               runtime_dispatch_address == 0x8065cce0u))
          {
            std::fprintf(stderr,
                         "[staticrecomp] register-trace after pc=%08x next=%08x "
                         "r0=%08x r1=%08x r3=%08x r4=%08x r5=%08x r7=%08x r8=%08x "
                         "r9=%08x lr=%08x cr=%08x\n",
                         runtime_dispatch_address, m_guest.pc, m_guest.gpr[0], m_guest.gpr[1],
                         m_guest.gpr[3], m_guest.gpr[4], m_guest.gpr[5], m_guest.gpr[7],
                         m_guest.gpr[8], m_guest.gpr[9], m_guest.lr, m_guest.cr);
          }
          if (has_rel_modules)
            m_guest.pc = TranslateRelAddress(m_guest.pc);
          if (selected_dispatch && sample_dispatch_time)
          {
            const TimePoint dispatch_time_end = Clock::now();
            const TimePoint dispatch_clock_end = Clock::now();
            m_dispatch_time_samples.push_back(
                {Common::FramePhaseTiming::GetPresentFrameIndex(),
                 Common::FramePhaseTiming::GetEmulatedFrameIndex(), sampled_dispatch_pc,
                 Common::FramePhaseTiming::ToNanoseconds(dispatch_time_end - dispatch_time_start),
                 Common::FramePhaseTiming::ToNanoseconds(dispatch_clock_end - dispatch_time_end)});
            if (m_dispatch_time_samples.size() >= 16384)
              FlushDispatchTimeSamples(false);
          }
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
          ppc.downcount -= static_cast<int>(charge > 0 ? charge : 1);
          m_charged_cycles += static_cast<u64>(charge > 0 ? charge : 1);
          // TB ticks once per TIMER_RATIO CPU cycles; derive from the SyncIn
          // snapshot so guest mftb stays monotonic across burst boundaries.
          m_burst_tb_cycles += static_cast<u64>(charge > 0 ? charge : 1);
          m_guest.timebase = m_burst_tb_base + m_burst_tb_cycles / SystemTimers::TIMER_RATIO;

          // Idle loop skipping for configured target loops (e.g. Wii Menu OSIdleThread)
          const bool configured_idle = idle_pc != 0 && m_guest.pc == idle_pc;
          const bool secondary_idle =
              secondary_idle_pc != 0 && m_guest.pc == secondary_idle_pc;
          const bool caller_idle = m_caller_idle_pc != 0 && m_caller_idle_lr != 0 &&
                                   m_guest.pc == m_caller_idle_pc &&
                                   m_guest.lr == m_caller_idle_lr;
          if (configured_idle || secondary_idle || caller_idle)
          {
            const bool canonical_boundary =
                m_caller_idle_pc != 0 && m_caller_idle_lr != 0 ? caller_idle : configured_idle;
            if (canonical_boundary && NetPlay::IsNetPlayRunning())
              CaptureNetplayBoundarySnapshot();
            const TimePoint idle_start = log_phase ? Clock::now() : TimePoint{};
            m_system.GetCoreTiming().Idle();
            if (caller_idle)
              ++m_caller_idle_hits;
            if (secondary_idle)
            {
              if (m_secondary_idle_hits == 0)
                std::fprintf(stderr, "[staticrecomp] secondary idle first hit pc=%08x\n",
                             secondary_idle_pc);
              ++m_secondary_idle_hits;
            }
            if (log_phase)
              Common::FramePhaseTiming::AddCpuIdle(Clock::now() - idle_start);
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
        } while (m_module_active && FastDispatchableAt(m_guest.pc) &&
                 !(m_guest.host_call && IsHostCallAddress(m_guest.pc)) && ppc.downcount > 0 &&
                 *state_ptr == CPU::State::Running);
        SyncOut();
        if (std::getenv("STATICRECOMP_REGISTER_TRACE") &&
            (ppc.pc < 0x10000u || ppc.Exceptions != 0 || m_guest.exception != 0))
        {
          std::fprintf(stderr,
                       "[staticrecomp] native-sync-out guest_pc=%08x ppc_pc=%08x npc=%08x "
                       "exceptions=%08x guest_exc=%08x r1=%08x r3=%08x r4=%08x "
                       "srr0=%08x srr1=%08x msr=%08x\n",
                       m_guest.pc, ppc.pc, ppc.npc, ppc.Exceptions, m_guest.exception,
                       ppc.gpr[1], ppc.gpr[3], ppc.gpr[4], ppc.spr[SPR_SRR0],
                       ppc.spr[SPR_SRR1], ppc.msr.Hex);
        }
        if ((ppc.Exceptions & SYNC_EXCEPTION_MASK) != 0)
        {
          if (std::getenv("STATICRECOMP_REGISTER_TRACE"))
            std::fprintf(stderr,
                         "[staticrecomp] native-check-exceptions-before pc=%08x npc=%08x exceptions=%08x\n",
                         ppc.pc, ppc.npc, ppc.Exceptions);
          power_pc.CheckExceptions();
          if (std::getenv("STATICRECOMP_REGISTER_TRACE"))
            std::fprintf(stderr,
                         "[staticrecomp] native-check-exceptions-after  pc=%08x npc=%08x exceptions=%08x\n",
                         ppc.pc, ppc.npc, ppc.Exceptions);
        }
      }
      else
      {
        if (std::getenv("STATICRECOMP_REGISTER_TRACE") &&
            (ppc.pc == 0x80343680u || ppc.pc == 0x8034c9b8u || ppc.pc == 0x803881c0u))
        {
          std::fprintf(stderr,
                       "[staticrecomp] register-trace fallback-before pc=%08x "
                       "r1=%08x r3=%08x r7=%08x r8=%08x r9=%08x lr=%08x cr=%08x npc=%08x\n",
                       ppc.pc, ppc.gpr[1], ppc.gpr[3], ppc.gpr[7], ppc.gpr[8], ppc.gpr[9],
                       LR(ppc), ppc.cr.Get(), ppc.npc);
        }
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
            ppc.downcount -= static_cast<int>(charge > 0 ? charge : 1);
            m_burst_tb_cycles += static_cast<u64>(charge > 0 ? charge : 1);
            m_guest.timebase = m_burst_tb_base + m_burst_tb_cycles / SystemTimers::TIMER_RATIO;
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
          ppc.downcount -= interpreter.SingleStepInner();
          ++m_fallback_steps;
        }
        else if (m_fallback_jit)
        {
          m_fallback_jit->Run();
        }
        else
        {
          if (std::getenv("STATICRECOMP_REGISTER_TRACE") &&
              (ppc.pc < 0x10000u || ppc.Exceptions != 0))
            std::fprintf(stderr,
                         "[staticrecomp] fallback-step pc=%08x npc=%08x exceptions=%08x\n",
                         ppc.pc, ppc.npc, ppc.Exceptions);
          do
          {
            ppc.downcount -= interpreter.SingleStepInner();
            ++m_fallback_steps;
          } while (!(m_module_active && DispatchableAt(ppc.pc)) &&
                   !IsHostCallAddress(ppc.pc) && ppc.downcount > 0 &&
                   *state_ptr == CPU::State::Running);
        }
      }
    } while (ppc.downcount > 0 && *state_ptr == CPU::State::Running);
    if (log_phase)
    {
      Common::FramePhaseTiming::AddCpuWall(Clock::now() - cpu_slice_start);
      Common::FramePhaseTiming::AddCpuThreadNanoseconds(GetCurrentThreadCpuNanoseconds() -
                                                        cpu_thread_start);
      Common::FramePhaseTiming::AddStaticRecompWork(
          m_bursts - phase_bursts, m_charged_cycles - phase_cycles,
          m_native_dispatches - phase_native_dispatches, m_fallback_steps - phase_fallback_steps,
          m_hook_fallback_instructions - phase_hook_fallbacks);
    }
  }
}

void StaticRecompCore::SingleStep()
{
  // Debugger stepping runs through the interpreter; state outside Run() lives
  // in PowerPCState, so no sync is needed.
  auto& system = m_system;
  system.GetCoreTiming().Advance();
  system.GetPPCState().downcount -= system.GetInterpreter().SingleStepInner();
}
