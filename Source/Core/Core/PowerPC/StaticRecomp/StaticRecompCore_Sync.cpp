// RecompCore: StaticRecomp CPU core - State synchronization.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/PowerPC/StaticRecomp/StaticRecompCore.h"
#include "Core/System.h"
#include "Core/Config/MainSettings.h"
#include "VideoCommon/Fifo.h"
#include "Core/PowerPC/PowerPC.h"
#include "Core/HW/SystemTimers.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
constexpr u64 FNV_OFFSET = 1469598103934665603ULL;

u64 FnvAppend(u64 hash, const void* data, size_t size)
{
  const auto* bytes = static_cast<const u8*>(data);
  for (size_t i = 0; i < size; ++i)
  {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}
}  // namespace

u64 StaticRecompCore::GetDiagnosticIntegerStateHash() const
{
  // Host callbacks, pointers, timebase, downcount, and cache bookkeeping are
  // excluded so the fingerprint is stable across native host architectures.
  u64 hash = FNV_OFFSET;
  const auto append = [&](const void* data, size_t size) { hash = FnvAppend(hash, data, size); };
  append(m_guest.gpr, sizeof(m_guest.gpr));
  append(&m_guest.pc, sizeof(m_guest.pc));
  append(&m_guest.lr, sizeof(m_guest.lr));
  append(&m_guest.ctr, sizeof(m_guest.ctr));
  append(&m_guest.cr, sizeof(m_guest.cr));
  append(&m_guest.xer, sizeof(m_guest.xer));
  append(&m_guest.msr, sizeof(m_guest.msr));
  append(&m_guest.srr0, sizeof(m_guest.srr0));
  append(&m_guest.srr1, sizeof(m_guest.srr1));
  append(&m_guest.dar, sizeof(m_guest.dar));
  append(&m_guest.dsisr, sizeof(m_guest.dsisr));
  append(&m_guest.ear, sizeof(m_guest.ear));
  append(&m_guest.hid2, sizeof(m_guest.hid2));
  append(m_guest.sr, sizeof(m_guest.sr));
  append(m_guest.gqr, sizeof(m_guest.gqr));
  append(&m_guest.exception, sizeof(m_guest.exception));
  append(&m_guest.program_exception, sizeof(m_guest.program_exception));
  append(&m_guest.reserve_addr, sizeof(m_guest.reserve_addr));
  append(&m_guest.reserve_valid, sizeof(m_guest.reserve_valid));
  return hash;
}

u64 StaticRecompCore::GetDiagnosticFprStateHash() const
{
  return FnvAppend(FNV_OFFSET, m_guest.fpr, sizeof(m_guest.fpr));
}

u64 StaticRecompCore::GetDiagnosticPairedStateHash() const
{
  u64 hash = FnvAppend(FNV_OFFSET, m_guest.ps1, sizeof(m_guest.ps1));
  return FnvAppend(hash, &m_guest.fpscr, sizeof(m_guest.fpscr));
}

u64 StaticRecompCore::GetDiagnosticStateHash() const
{
  u64 hash = FNV_OFFSET;
  const u64 parts[] = {GetDiagnosticIntegerStateHash(), GetDiagnosticFprStateHash(),
                       GetDiagnosticPairedStateHash()};
  return FnvAppend(hash, parts, sizeof(parts));
}

u64 StaticRecompCore::HashSelectedRamPages() const
{
  // Sample the first cache line of every 4 KiB MEM1 page. This spans the full
  // memory allocation while keeping a once-per-second diagnostic far below a
  // full 24 MiB scan on mobile CPUs.
  constexpr u32 PAGE_SIZE = 4096;
  constexpr u32 SAMPLE_SIZE = 64;
  u64 hash = FNV_OFFSET;
  for (u32 offset = 0; offset < m_guest.ram_size; offset += PAGE_SIZE)
  {
    const u32 remaining = m_guest.ram_size - offset;
    const u32 size = remaining < SAMPLE_SIZE ? remaining : SAMPLE_SIZE;
    hash = FnvAppend(hash, m_guest.ram + offset, size);
  }
  return hash;
}

std::array<u64, 32> StaticRecompCore::HashSelectedRamRegions() const
{
  constexpr u32 REGION_SIZE = 1024 * 1024;
  constexpr u32 PAGE_SIZE = 4096;
  constexpr u32 SAMPLE_SIZE = 64;
  std::array<u64, 32> hashes;
  hashes.fill(FNV_OFFSET);
  for (u32 offset = 0; offset < m_guest.ram_size; offset += PAGE_SIZE)
  {
    const size_t region = offset / REGION_SIZE;
    if (region >= hashes.size())
      break;
    const u32 remaining = m_guest.ram_size - offset;
    const u32 size = remaining < SAMPLE_SIZE ? remaining : SAMPLE_SIZE;
    hashes[region] = FnvAppend(hashes[region], m_guest.ram + offset, size);
  }
  return hashes;
}

void StaticRecompCore::CaptureNetplayBoundarySnapshot()
{
  ++m_netplay_boundary_sequence;
  if (m_netplay_boundary_sequence % 60 != 0)
    return;

  if (std::getenv("MELEEPAD_NETPLAY_TRACE_BOUNDARY_CLOCK"))
  {
    if (m_netplay_boundary_sequence == 60)
    {
      std::fprintf(stderr,
                   "[staticrecomp] netplay-scheduling dual_core=%u deterministic_gpu=%u "
                   "sync_gpu=%u sync_on_idle=%u max_distance=%d min_distance=%d "
                   "overclock_enabled=%u overclock=%.6f\n",
                   unsigned(m_system.IsDualCoreMode()),
                   unsigned(m_system.GetFifo().UseDeterministicGPUThread()),
                   unsigned(m_system.GetFifo().UseSyncGPU()),
                   unsigned(Config::Get(Config::MAIN_SYNC_ON_SKIP_IDLE)),
                   Config::Get(Config::MAIN_SYNC_GPU_MAX_DISTANCE),
                   Config::Get(Config::MAIN_SYNC_GPU_MIN_DISTANCE),
                   unsigned(Config::Get(Config::MAIN_OVERCLOCK_ENABLE)),
                   double(Config::Get(Config::MAIN_OVERCLOCK)));
    }
    const u64 live_timebase = m_system.GetSystemTimers().GetFakeTimeBase();
    std::fprintf(stderr,
                 "[staticrecomp] boundary-clock sequence=%llu cached=%llu live=%llu "
                 "base=%llu cycles=%llu downcount=%d\n",
                 static_cast<unsigned long long>(m_netplay_boundary_sequence),
                 static_cast<unsigned long long>(m_guest.timebase),
                 static_cast<unsigned long long>(live_timebase),
                 static_cast<unsigned long long>(m_burst_tb_base),
                 static_cast<unsigned long long>(m_burst_tb_cycles),
                 m_system.GetPPCState().downcount);
  }

  m_netplay_boundary_snapshot = {
      .sequence = m_netplay_boundary_sequence,
      .guest_pc = m_guest.pc,
      .timebase = m_guest.timebase,
      .state_hash = GetDiagnosticStateHash(),
      .integer_state_hash = GetDiagnosticIntegerStateHash(),
      .fpr_state_hash = GetDiagnosticFprStateHash(),
      .paired_state_hash = GetDiagnosticPairedStateHash(),
      .ram_hash = HashSelectedRamPages(),
      .ram_region_hashes = HashSelectedRamRegions(),
  };
  if (m_netplay_boundary_sequence == 60)
  {
    std::fprintf(stderr, "[staticrecomp] canonical-boundary active pc=%08x\n",
                 m_netplay_boundary_snapshot.guest_pc);
  }
}

void StaticRecompCore::SetPPCStateFromGuestState(const CPUState& s, PowerPC::PowerPCState& ppc)
{
  std::memcpy(ppc.gpr, s.gpr, sizeof(ppc.gpr));
  for (int i = 0; i < 32; ++i)
  {
    std::memcpy(&ppc.ps[i].ps0, &s.fpr[i], sizeof(u64));
    std::memcpy(&ppc.ps[i].ps1, &s.ps1[i], sizeof(u64));
  }
  ppc.pc = s.pc;
  ppc.npc = s.pc;
  ppc.spr[SPR_LR] = s.lr;
  ppc.spr[SPR_CTR] = s.ctr;
  ppc.cr.Set(s.cr);
  ppc.SetXER(UReg_XER{s.xer});
  ppc.fpscr.Hex = s.fpscr;
  ppc.spr[SPR_SRR0] = s.srr0;
  ppc.spr[SPR_SRR1] = s.srr1;
  ppc.spr[SPR_DAR] = s.dar;
  ppc.spr[SPR_DSISR] = s.dsisr;
  ppc.spr[SPR_EAR] = s.ear;
  ppc.spr[SPR_HID2] = s.hid2;
  for (int i = 0; i < 16; ++i)
    ppc.sr[i] = s.sr[i];
  for (int i = 0; i < 8; ++i)
    ppc.spr[SPR_GQR0 + i] = s.gqr[i];
  ppc.reserve_address = s.reserve_addr;
  ppc.reserve = s.reserve_valid;
}

void StaticRecompCore::SyncIn()
{
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();

  if (std::getenv("STATICRECOMP_REGISTER_TRACE") && ppc.pc == 0)
  {
    std::fprintf(stderr,
                 "[staticrecomp] register-trace sync-in-zero pc=%08x npc=%08x "
                 "r1=%08x r3=%08x r4=%08x r7=%08x lr=%08x cr=%08x srr0=%08x srr1=%08x msr=%08x\n",
                 ppc.pc, ppc.npc, ppc.gpr[1], ppc.gpr[3], ppc.gpr[4], ppc.gpr[7],
                 ppc.spr[SPR_LR], ppc.cr.Get(), ppc.spr[SPR_SRR0], ppc.spr[SPR_SRR1],
                 ppc.msr.Hex);
  }

  std::memcpy(m_guest.gpr, ppc.gpr, sizeof(m_guest.gpr));
  for (int i = 0; i < 32; ++i)
  {
    std::memcpy(&m_guest.fpr[i], &ppc.ps[i].ps0, sizeof(u64));
    std::memcpy(&m_guest.ps1[i], &ppc.ps[i].ps1, sizeof(u64));
  }
  m_guest.pc = ppc.pc;
  m_guest.lr = ppc.spr[SPR_LR];
  m_guest.ctr = ppc.spr[SPR_CTR];
  m_guest.cr = ppc.cr.Get();
  m_guest.xer = ppc.GetXER().Hex;
  m_guest.fpscr = ppc.fpscr.Hex;
  m_guest.msr = ppc.msr.Hex;
  m_guest.srr0 = ppc.spr[SPR_SRR0];
  m_guest.srr1 = ppc.spr[SPR_SRR1];
  m_guest.dar = ppc.spr[SPR_DAR];
  m_guest.dsisr = ppc.spr[SPR_DSISR];
  m_guest.ear = ppc.spr[SPR_EAR];
  m_guest.hid2 = ppc.spr[SPR_HID2];
  for (int i = 0; i < 16; ++i)
    m_guest.sr[i] = ppc.sr[i];
  for (int i = 0; i < 8; ++i)
    m_guest.gqr[i] = ppc.spr[SPR_GQR0 + i];
  // Dolphin materializes TB lazily on read (spr[TL/TU] is a stale cache);
  // GetFakeTimeBase() is the live value, matching the interpreter's mftb.
  m_guest.timebase = m_system.GetSystemTimers().GetFakeTimeBase();
  m_burst_tb_base = m_guest.timebase;
  m_burst_tb_cycles = 0;
  m_guest.reserve_addr = ppc.reserve_address;
  m_guest.reserve_valid = ppc.reserve;
  m_guest.exception = 0;
  m_guest.program_exception = 0;
  m_guest.downcount = 0;  // charge accumulator, not a copy of ppc.downcount

  if (m_module && m_module->on_state_loaded)
    m_module->on_state_loaded(&m_guest);
}

void StaticRecompCore::SyncOut()
{
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();

  if (std::getenv("STATICRECOMP_REGISTER_TRACE") && m_guest.pc == 0)
  {
    std::fprintf(stderr,
                 "[staticrecomp] register-trace sync-out-zero guest_pc=%08x ppc_pc=%08x "
                 "r1=%08x r3=%08x r4=%08x r7=%08x lr=%08x cr=%08x srr0=%08x srr1=%08x msr=%08x\n",
                 m_guest.pc, ppc.pc, m_guest.gpr[1], m_guest.gpr[3], m_guest.gpr[4],
                 m_guest.gpr[7], m_guest.lr, m_guest.cr, m_guest.srr0, m_guest.srr1,
                 m_guest.msr);
  }

  // Flush any cycle charge the module accumulated since the last flush
  // (matters on the mid-dispatch fallback path, where the interpreter step
  // that follows must see up-to-date time).
  ppc.downcount += static_cast<int>(m_guest.downcount);
  m_guest.downcount = 0;

  SetPPCStateFromGuestState(m_guest, ppc);
  // Timebase is owned by Dolphin's CoreTiming; native code cannot write it.

  if (ppc.msr.Hex != m_guest.msr)
  {
    ppc.msr.Hex = m_guest.msr;
    power_pc.MSRUpdated();
  }
  PowerPC::RoundingModeUpdated(ppc);
}

void StaticRecompCore::PropagateGuestMSR()
{
  auto& power_pc = m_system.GetPowerPC();
  auto& ppc = power_pc.GetPPCState();
  if (ppc.msr.Hex != m_guest.msr)
  {
    ppc.msr.Hex = m_guest.msr;
    power_pc.MSRUpdated();
  }
}
