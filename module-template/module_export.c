// RecompCore per-game native module export glue (game id set at build time).
//
// Wraps the DolRecomp-generated constant-time chunk dispatcher behind the
// StaticRecomp module ABI. All environment access goes through the CPUState
// hook pointers the chassis installs; this dylib has no host dependencies.

#include "generated.h"

#include "StaticRecompABI.h"

#if defined(GALAXYPAD_OUTLINE_DISPATCH_SLOWPATH) && GALAXYPAD_OUTLINE_DISPATCH_SLOWPATH && \
    !defined(DOLRECOMP_ENABLE_REPLACEMENTS)
// Keep hook and physical-address handling in the original implementation.
// The common virtual-address path needs neither observer calls nor alias retry.
static __attribute__((noinline)) int chassis_dispatch_slow(CPUState* ctx, u32 address)
{
    return dolrecomp_call(ctx, address);
}
#endif

static int chassis_dispatch(CPUState* ctx, u32 address)
{
#if defined(GALAXYPAD_OUTLINE_DISPATCH_SLOWPATH) && GALAXYPAD_OUTLINE_DISPATCH_SLOWPATH && \
    !defined(DOLRECOMP_ENABLE_REPLACEMENTS)
    if (ctx->host_call)
        return chassis_dispatch_slow(ctx, address);
    ctx->pc = address;
    DolRecompFunction fn = dolrecomp_find_original(address);
    if (!fn)
        return chassis_dispatch_slow(ctx, address);
    fn(ctx);
    return 1;
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

#if defined(GALAXYPAD_VOID_DISPATCH) && GALAXYPAD_VOID_DISPATCH
// Optional v1 entry for callers that do not consume dispatch's coverage result.
// Keep the original ABI entry unchanged. A void return lets the final chunk
// call be a tail call; hook ordering and physical-address retries stay exact.
RECOMP_MODULE_EXPORT void staticrecomp_dispatch_void_v1(CPUState* ctx, u32 address)
{
    u32 alias;
    ctx->pc = address;
    if (dolrecomp_dispatch_replacement(ctx, address)) return;
    if (ctx->host_call && ppc_host_call(ctx, address)) return;
    DolRecompFunction fn = dolrecomp_find_original(address);
    if (fn) {
        ctx->pc = address;
        fn(ctx);
        return;
    }
    if (dolrecomp_physical_pc_alias(ctx, address, &alias)) {
        ctx->pc = alias;
        if (dolrecomp_dispatch_replacement(ctx, alias)) return;
        if (ctx->host_call && ppc_host_call(ctx, alias)) return;
        fn = dolrecomp_find_original(alias);
        if (fn) {
            ctx->pc = alias;
            fn(ctx);
        }
    }
}
#endif
