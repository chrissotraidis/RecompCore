// RecompCore per-game native module export glue (game id set at build time).
//
// Wraps the DolRecomp-generated constant-time chunk dispatcher behind the
// StaticRecomp module ABI. All environment access goes through the CPUState
// hook pointers the chassis installs; this dylib has no host dependencies.

#include "generated.h"

#include "StaticRecompABI.h"

#ifdef RECOMPCORE_MODULE_DISPATCH_COUNTS
#include <stdio.h>
#include <string.h>

typedef struct DispatchCounts
{
    u64 dispatches;
    u64 replacement_hits;
    u64 host_probes;
    u64 host_hits;
    u64 original_hits;
    u64 alias_attempts;
    u64 alias_replacement_hits;
    u64 alias_host_probes;
    u64 alias_host_hits;
    u64 alias_original_hits;
    u64 misses;
} DispatchCounts;

static DispatchCounts s_dispatch_counts;

static int counted_dispatch(CPUState* ctx, u32 address)
{
    u32 alias;
    s_dispatch_counts.dispatches++;
    ctx->pc = address;
    if (dolrecomp_dispatch_replacement(ctx, address))
    {
        s_dispatch_counts.replacement_hits++;
        return 1;
    }
    if (ctx->host_call)
    {
        s_dispatch_counts.host_probes++;
        if (ppc_host_call(ctx, address))
        {
            s_dispatch_counts.host_hits++;
            return 1;
        }
    }
    if (dolrecomp_call_original(ctx, address))
    {
        s_dispatch_counts.original_hits++;
        return 1;
    }
    if (dolrecomp_physical_pc_alias(ctx, address, &alias))
    {
        s_dispatch_counts.alias_attempts++;
        ctx->pc = alias;
        if (dolrecomp_dispatch_replacement(ctx, alias))
        {
            s_dispatch_counts.alias_replacement_hits++;
            return 1;
        }
        if (ctx->host_call)
        {
            s_dispatch_counts.alias_host_probes++;
            if (ppc_host_call(ctx, alias))
            {
                s_dispatch_counts.alias_host_hits++;
                return 1;
            }
        }
        if (dolrecomp_call_original(ctx, alias))
        {
            s_dispatch_counts.alias_original_hits++;
            return 1;
        }
    }
    s_dispatch_counts.misses++;
    return 0;
}
#endif

#ifdef __LLVM_INSTR_PROFILE_GENERATE
#include <profile/instr_prof_interface.h>
#endif

static int chassis_dispatch(CPUState* ctx, u32 address)
{
#ifdef RECOMPCORE_MODULE_DISPATCH_COUNTS
    return counted_dispatch(ctx, address);
#else
    return dolrecomp_call(ctx, address);
#endif
}

static void chassis_on_state_loaded(CPUState* ctx)
{
    // Re-arm host FP rounding/flush state from the freshly loaded guest FPSCR.
    ppc_fpscr_updated(ctx);
}

#include "module_tables.inc"

static const StaticRecompModuleDesc s_desc = {
    STATICRECOMP_ABI_VERSION,
    GXRUNTIME_CPU_ABI_VERSION,
    (u32)sizeof(CPUState),
    MODULE_GAME_ID,
    DOLRECOMP_ENTRY_POINT,
    chassis_dispatch,
    chassis_on_state_loaded,
    s_code_ranges,
    MODULE_CODE_RANGE_COUNT,
    s_smc_ranges,
    MODULE_SMC_RANGE_COUNT,
    s_chunk_ranges,
    MODULE_CHUNK_RANGE_COUNT,
    s_chunk_hashes,
};

#if defined(_WIN32)
#define RECOMP_MODULE_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define RECOMP_MODULE_EXPORT __attribute__((visibility("default")))
#else
#define RECOMP_MODULE_EXPORT
#endif

RECOMP_MODULE_EXPORT const StaticRecompModuleDesc* staticrecomp_get_module(void)
{
    return &s_desc;
}

#ifdef __LLVM_INSTR_PROFILE_GENERATE
RECOMP_MODULE_EXPORT void staticrecomp_profile_reset(void)
{
    __llvm_profile_reset_counters();
}

RECOMP_MODULE_EXPORT int staticrecomp_profile_dump(void)
{
    return __llvm_profile_dump();
}
#elif defined(RECOMPCORE_MODULE_DISPATCH_COUNTS)
RECOMP_MODULE_EXPORT void staticrecomp_profile_reset(void)
{
    memset(&s_dispatch_counts, 0, sizeof(s_dispatch_counts));
}

RECOMP_MODULE_EXPORT int staticrecomp_profile_dump(void)
{
    fprintf(stderr,
            "[dispatch-counts] dispatches=%llu replacement_hits=%llu "
            "host_probes=%llu host_hits=%llu original_hits=%llu "
            "alias_attempts=%llu alias_replacement_hits=%llu "
            "alias_host_probes=%llu alias_host_hits=%llu "
            "alias_original_hits=%llu misses=%llu\n",
            (unsigned long long)s_dispatch_counts.dispatches,
            (unsigned long long)s_dispatch_counts.replacement_hits,
            (unsigned long long)s_dispatch_counts.host_probes,
            (unsigned long long)s_dispatch_counts.host_hits,
            (unsigned long long)s_dispatch_counts.original_hits,
            (unsigned long long)s_dispatch_counts.alias_attempts,
            (unsigned long long)s_dispatch_counts.alias_replacement_hits,
            (unsigned long long)s_dispatch_counts.alias_host_probes,
            (unsigned long long)s_dispatch_counts.alias_host_hits,
            (unsigned long long)s_dispatch_counts.alias_original_hits,
            (unsigned long long)s_dispatch_counts.misses);
    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((destructor))
#endif
static void dump_dispatch_counts_on_unload(void)
{
    if (s_dispatch_counts.dispatches != 0)
        (void)staticrecomp_profile_dump();
}
#endif
