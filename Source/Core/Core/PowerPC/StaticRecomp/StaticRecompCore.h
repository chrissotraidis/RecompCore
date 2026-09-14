// RecompCore: StaticRecomp CPU core.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Common/CommonTypes.h"
#include "Common/DynamicLibrary.h"
#include "Core/PowerPC/JitCommon/JitBase.h"
#include "Core/PowerPC/JitCommon/JitCache.h"
#include "Core/PowerPC/StaticRecomp/StaticRecompABI.h"
#include "Core/PowerPC/StaticRecomp/StaticRecompModuleSource.h"

namespace Core
{
class System;
}

namespace PowerPC
{
struct PowerPCState;
}

namespace StaticRecompLockstep
{
class StaticRecompLockstepVerifier;
}

// Executes statically recompiled per-game native code when the PC is covered by
// a loaded module; falls back to Dolphin's interpreter for everything else.
// With no module loaded this core is exactly an interpreter loop.
class StaticRecompCore : public JitBase
{
public:
  friend class StaticRecompLockstep::StaticRecompLockstepVerifier;

  explicit StaticRecompCore(Core::System& system, StaticRecompModuleSource module_source);
  StaticRecompCore(const StaticRecompCore&) = delete;
  StaticRecompCore(StaticRecompCore&&) = delete;
  StaticRecompCore& operator=(const StaticRecompCore&) = delete;
  StaticRecompCore& operator=(StaticRecompCore&&) = delete;
  ~StaticRecompCore() override;

  void Init() override;
  void Shutdown() override;

  void Run() override;
  void SingleStep() override;
  bool IsModuleActive() const;
  bool DispatchableAt(u32 address);
  bool FastDispatchableAt(u32 address);
  bool IsHostCallAddress(u32 address) const;
  bool ShouldYieldAt(u32 address);

  // Read-only counters sampled by netplay desync diagnostics on the CPU thread.
  u32 GetDiagnosticGuestPC() const { return m_guest.pc; }
  u64 GetDiagnosticStateHash() const;
  u64 GetDiagnosticIntegerStateHash() const;
  u64 GetDiagnosticFprStateHash() const;
  u64 GetDiagnosticPairedStateHash() const;
  u64 GetDiagnosticNativeDispatches() const { return m_native_dispatches; }
  u64 GetDiagnosticFallbackSteps() const { return m_fallback_steps; }
  u32 GetDiagnosticFailedChunks() const { return m_failed_chunks; }
  u64 GetDiagnosticVerifications() const { return m_verifications; }
  u64 GetDiagnosticReverifyEvents() const { return m_reverify_events; }
  u64 GetDiagnosticChargedCycles() const { return m_charged_cycles; }
  u64 GetDiagnosticBursts() const { return m_bursts; }

  struct NetplayBoundarySnapshot
  {
    u64 sequence = 0;
    u32 guest_pc = 0;
    u64 timebase = 0;
    u64 state_hash = 0;
    u64 integer_state_hash = 0;
    u64 fpr_state_hash = 0;
    u64 paired_state_hash = 0;
    u64 ram_hash = 0;
    std::array<u64, 32> ram_region_hashes{};
  };
  NetplayBoundarySnapshot GetNetplayBoundarySnapshot() const
  {
    return m_netplay_boundary_snapshot;
  }

  void ClearCache() override;
  void Jit(u32 em_address) override {}
  bool HandleFault(uintptr_t access_address, SContext* ctx) override { return false; }

  JitBaseBlockCache* GetBlockCache() override { return &m_block_cache; }
  void EraseSingleBlock(const JitBlock& block) override {}
  std::vector<MemoryStats> GetMemoryStats() const override { return {}; }
  std::size_t DisassembleNearCode(const JitBlock& block, std::ostream& stream) const override
  {
    return 0;
  }
  std::size_t DisassembleFarCode(const JitBlock& block, std::ostream& stream) const override
  {
    return 0;
  }
  const CommonAsmRoutinesBase* GetAsmRoutines() override { return nullptr; }
  const char* GetName() const override { return "StaticRecomp"; }

private:
  template <bool Diagnostics>
  void RunWithDiagnostics();
  // JitBaseBlockCache with no generated blocks; exists so generic
  // icache-invalidation plumbing in JitInterface has a real object to talk
  // to, and to feed every invalidation into the SMC demotion guard (D4).
  class EmptyBlockCache : public JitBaseBlockCache
  {
  public:
    explicit EmptyBlockCache(StaticRecompCore& core) : JitBaseBlockCache(core), m_core(core) {}
    void WriteLinkBlock(const JitBlock::LinkData& source, const JitBlock* dest) override {}

  protected:
    void InvalidateICacheInternal(u32 physical_address, u32 address, u32 length,
                                  bool forced) override
    {
      m_core.OnICacheInvalidate(address, length);
    }

  private:
    StaticRecompCore& m_core;
  };

  void LoadModule();
  void UpdateProfileCapture();
  void CaptureNetplayBoundarySnapshot();
  u64 HashSelectedRamPages() const;
  std::array<u64, 32> HashSelectedRamRegions() const;
  void FlushDispatchFrameSamples(bool final);
  void FlushDispatchTimeSamples(bool final);
  void FlushDispatchBurstSamples(bool final);

  // D4 SMC guard, verify-on-entry model. Every chunk starts Unverified; the
  // first native dispatch into it hashes its guest RAM against the module's
  // recorded hash of the original text. An icache invalidation touching a
  // chunk resets it to Unverified (Dolphin invalidates while *loading* code,
  // so invalidation alone must not retire coverage); a hash mismatch (real
  // SMC) marks it Failed, interpreter-only until the next invalidation.
  enum ChunkState : u8
  {
    CHUNK_UNVERIFIED = 0,
    CHUNK_VERIFIED = 1,
    CHUNK_FAILED = 2,
  };

  void OnICacheInvalidate(u32 address, u32 length);
  int ChunkIndexOf(u32 address);
  bool IsForcedFallbackAddress(u32 address) const;
  bool ChunkContainsHostCall(u32 index) const;
  void VerifyChunk(u32 index);
  bool ResolveNativeAddress(u32 runtime_address, u32* linked_address, u32* rel_section_index);
  bool ResolveRuntimeAddress(u32 linked_address, u32* runtime_address) const;
  u32 TranslateRelAddress(u32 linked_address);
  void RefreshRelSections();

  static void SetPPCStateFromGuestState(const CPUState& s, PowerPC::PowerPCState& ppc);

  // D1 state residency: registers live in m_guest while native code runs;
  // full sync at every native-burst boundary.
  void SyncIn();   // Dolphin PowerPCState -> m_guest
  void SyncOut();  // m_guest -> Dolphin PowerPCState

  // CPUState hooks (module -> chassis environment). `cpu->external_user_data`
  // is the StaticRecompCore*.
  static u64 HookExternalRead(CPUState* cpu, u32 ea, u8 size);
  static void HookExternalWrite(CPUState* cpu, u32 ea, u64 value, u8 size);
  static u32 HookExternalRead32(CPUState* cpu, u32 ea, u8 rid);
  static void HookExternalWrite32(CPUState* cpu, u32 ea, u32 value, u8 rid);
  static void* HookExternalPointer(CPUState* cpu, u32 ea, u32 size);
  static u32 HookSPRRead(CPUState* cpu, u16 spr, u32 cia);
  static void HookSPRWrite(CPUState* cpu, u16 spr, u32 value, u32 cia);
  static void HookCacheControl(CPUState* cpu, u8 operation, u32 ea, u32 cia);
  static void HookInstructionFallback(CPUState* cpu, u32 raw, u32 cia);
  static bool HookHostCall(CPUState* cpu, u32 address);

  // Keep Dolphin's MSR-derived state (translation mode, feature flags) in step
  // with the guest MSR before any MMU access or exception delivery.
  void PropagateGuestMSR();

  std::unique_ptr<StaticRecompLockstep::StaticRecompLockstepVerifier> m_lockstep_verifier;

  EmptyBlockCache m_block_cache{*this};

  CPUState m_guest{};
  Common::DynamicLibrary m_library;
  using ProfileResetFn = void (*)();
  using ProfileDumpFn = int (*)();
  ProfileResetFn m_profile_reset = nullptr;
  ProfileDumpFn m_profile_dump = nullptr;
  u32 m_profile_trigger_address = 0;
  u32 m_profile_trigger_mask = 0;
  u32 m_profile_trigger_value = 0;
  bool m_profile_capture_active = false;
  bool m_profile_capture_complete = false;
  StaticRecompModuleSource m_module_source;
  const StaticRecompModuleDesc* m_module = nullptr;
  bool m_module_active = false;
  u32 m_host_call_passthrough_pc = 0;
  bool m_host_call_passthrough = false;
  std::unique_ptr<JitBase> m_fallback_jit;

  u64 m_native_dispatches = 0;
  u64 m_fallback_steps = 0;
  u64 m_native_exceptions = 0;
  u64 m_hook_fallback_instructions = 0;
  std::unordered_map<u32, u64> m_dispatch_samples;
  struct DispatchFrameSample
  {
    u64 frame;
    u32 pc;
  };
  std::string m_dispatch_frame_log_path;
  std::vector<DispatchFrameSample> m_dispatch_frame_samples;
  bool m_dispatch_frame_log_started = false;
  u64 m_dispatch_frame_samples_written = 0;
  u64 m_dispatch_sample_interval = 4096;
  u64 m_dispatch_sample_offset = 0;
  struct DispatchTimeSample
  {
    u64 present_frame;
    u64 emulated_frame;
    u32 pc;
    u64 host_ns;
    u64 clock_ns;
  };
  std::string m_dispatch_time_log_path;
  std::vector<DispatchTimeSample> m_dispatch_time_samples;
  bool m_dispatch_time_log_started = false;
  u64 m_dispatch_time_samples_written = 0;
  struct DispatchBurstSample
  {
    u64 present_frame;
    u64 emulated_frame;
    u64 burst;
    u32 index;
    u32 previous_pc;
    u32 pc;
  };
  std::string m_dispatch_burst_log_path;
  std::vector<DispatchBurstSample> m_dispatch_burst_samples;
  bool m_dispatch_burst_log_started = false;
  u64 m_dispatch_burst_samples_written = 0;
  u64 m_dispatch_burst_id = 0;
  u32 m_dispatch_burst_index = 0;
  u32 m_dispatch_burst_remaining = 0;
  u32 m_dispatch_burst_previous_pc = 0;
  u64 m_bursts = 0;          // SyncIn..SyncOut native runs (diagnostic)
  u64 m_charged_cycles = 0;  // cycles flushed from module charges (diagnostic)

  // ctx->timebase is in TB ticks (1 tick per SystemTimers::TIMER_RATIO CPU
  // cycles), while module charges are CPU cycles. Deriving in-burst TB from a
  // SyncIn snapshot plus accumulated cycles keeps guest mftb monotonic and in
  // agreement with GetFakeTimeBase() at burst boundaries.
  u64 m_burst_tb_base = 0;    // GetFakeTimeBase() at the last SyncIn
  u64 m_burst_tb_cycles = 0;  // CPU cycles charged since that snapshot

  // D4 guard state: parallel to m_module->chunk_ranges.
  std::vector<u8> m_chunk_state;
  mutable std::vector<u8> m_chunk_host_call_state;
  std::vector<StaticRecompRange> m_forced_fallback_ranges;
  struct ActiveRelSection
  {
    u32 module_id;
    u32 section_index;
    u32 linked_start;
    u32 runtime_start;
    u32 size;
  };
  std::vector<ActiveRelSection> m_active_rel_sections;
  std::vector<int> m_chunk_rel_sections;
  std::vector<u64> m_effective_chunk_hashes;
  u64 m_rel_mapping_generation = 0;
  u32 m_failed_chunks = 0;    // chunks currently failing verification (real SMC)
  u64 m_verifications = 0;    // chunk hash checks performed
  u64 m_reverify_events = 0;  // invalidations that reset a chunk to Unverified

  // Lookup table optimization for O(1) chunk searches
  std::vector<int> m_chunk_lookup_table;
  u32 m_lookup_ram_size = 0;
  u32 m_lookup_exram_size = 0;
  int GetAddressLookupIndex(u32 address) const;
  void InitLookupTable(u32 ram_size, u32 exram_size);

  // Dispatch locality: most control transfers stay inside one chunk, so the
  // last hit short-circuits the chunk binary search on the hot path.
  mutable u32 m_last_chunk_index = 0;

  u32 m_idle_pc = 0;
  u32 m_secondary_idle_pc = 0;
  u64 m_secondary_idle_hits = 0;
  u32 m_caller_idle_pc = 0;
  u32 m_caller_idle_lr = 0;
  u64 m_caller_idle_hits = 0;
  u64 m_netplay_boundary_sequence = 0;
  NetplayBoundarySnapshot m_netplay_boundary_snapshot{};
};

extern StaticRecompCore* g_static_recomp_core;
u32 StaticRecompShouldYieldAt(u32 address);
