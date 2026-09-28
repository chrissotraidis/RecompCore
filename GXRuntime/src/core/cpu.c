// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <stdatomic.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define GXRUNTIME_EXPORT __attribute__((visibility("default")))
#else
#define GXRUNTIME_EXPORT
#endif

/* --- Chassis lockstep journal (debug-only, NOT part of the CPU ABI) ---------
 * Optional callback fired just before every committed RAM write. The
 * dolphin-chassis differential ("lockstep") harness installs it to snapshot
 * pre-images, so it can restore a pre-block memory view before re-running a
 * block on Dolphin's interpreter (correct read-modify-write comparison).
 * NULL and zero-cost unless installed; the chassis resolves the setter by name
 * via dlsym, so its absence simply disables lockstep. `offset` is the RAM byte
 * offset (into cpu->ram) about to be written, `size` the width in bytes. */
PPCMemWriteJournal g_mem_write_journal = NULL;
void* g_mem_write_journal_user = NULL;
__attribute__((visibility("default"))) void ppc_set_mem_write_journal(PPCMemWriteJournal fn, void* user) {
    g_mem_write_journal = fn;
    g_mem_write_journal_user = user;
}

#define PPC_GUEST_ALIAS_MAX 4096u

typedef struct PPCGuestAlias {
    u32 linked_start;
    u32 size;
    u8* storage;
    bool owns_storage;
} PPCGuestAlias;

static PPCGuestAlias g_guest_aliases[PPC_GUEST_ALIAS_MAX];
static u32 g_guest_alias_count;
static u32 g_guest_alias_min_start = UINT32_MAX;
static u64 g_guest_alias_max_end;
bool g_ppc_guest_aliases_overlap_mem1;

// Sorted index over the registry, for the resolver.
//
// The resolver used to scan the registry linearly, and Wind Waker's linked REL
// data registers about 1,900 aliases, so an access inside the aliased range
// walked hundreds of entries and a miss walked all of them: 5.7% of the game
// thread in an Outset play-scene sample. The scan answers "the first alias, in
// registration order, that contains the access". When no two aliases overlap
// at most one can contain it, so a binary search over start addresses gives the
// same answer. The index is kept sorted by the mutators (never rebuilt lazily
// by a reader), and any overlap sends the resolver back to the scan.
static u32 g_guest_alias_sorted[PPC_GUEST_ALIAS_MAX];
static bool g_guest_alias_overlapping;

static bool ppc_guest_alias_pair_overlaps(const PPCGuestAlias* a,
                                          const PPCGuestAlias* b) {
    return (u64)a->linked_start < (u64)b->linked_start + b->size &&
           (u64)b->linked_start < (u64)a->linked_start + a->size;
}

static void ppc_guest_alias_index_insert(u32 index) {
    const PPCGuestAlias* alias = &g_guest_aliases[index];
    const u32 sorted_count = index; // entries [0, index) are already indexed
    u32 lo = 0u, hi = sorted_count;
    while (lo < hi) { // first position whose start is greater (upper bound)
        const u32 mid = lo + (hi - lo) / 2u;
        if (g_guest_aliases[g_guest_alias_sorted[mid]].linked_start <=
            alias->linked_start)
            lo = mid + 1u;
        else
            hi = mid;
    }
    if (lo < sorted_count)
        memmove(&g_guest_alias_sorted[lo + 1u], &g_guest_alias_sorted[lo],
                (sorted_count - lo) * sizeof(g_guest_alias_sorted[0]));
    g_guest_alias_sorted[lo] = index;
    if (lo > 0u &&
        ppc_guest_alias_pair_overlaps(
            &g_guest_aliases[g_guest_alias_sorted[lo - 1u]], alias))
        g_guest_alias_overlapping = true;
    if (lo + 1u < sorted_count + 1u &&
        ppc_guest_alias_pair_overlaps(
            alias, &g_guest_aliases[g_guest_alias_sorted[lo + 1u]]))
        g_guest_alias_overlapping = true;
}

static void ppc_guest_alias_index_rebuild(void) {
    g_guest_alias_overlapping = false;
    for (u32 i = 0u; i < g_guest_alias_count; ++i)
        ppc_guest_alias_index_insert(i);
}

// Guest-alias generation, for readers that cache a resolved host pointer.
//
// The host's per-block overlap observation caches the pointer it resolved for a
// guest address and revalidates it when the aliasing state moves. It used to ask
// that by reading the boolean above, which cost a load, a select into 0/1, a
// second load for the cached copy and a two-way compare at every block boundary
// (docs/status/CURRENT.md, 2026-09-22). A counter that moves whenever the alias
// set is touched answers the same question with one load and one compare, and it
// cannot be missed by a write site that updates the boolean without bumping it -
// every write goes through the setter below.
u32 g_ppc_guest_alias_generation;

static void ppc_guest_alias_set_overlap_mem1(bool overlaps) {
    g_ppc_guest_aliases_overlap_mem1 = overlaps;
    g_ppc_guest_alias_generation++;
}
// Cost census for the alias path (docs/status/REORGANIZATION_2026-09-17.md, W4).
// get_ram_ptr sends every untagged MEM1 access through ppc_guest_alias_resolve
// as soon as one alias touches MEM1, and the resolver is a linear scan of the
// registry. Whether that matters depends on numbers nobody has: how often the
// resolver is entered, how far it scans before it answers, and how often a scan
// is cut short by the bounds test. Inert unless BLUEWAKE_TRACE_ALIAS_COST is set.
static u64 g_alias_cost_calls;
static u64 g_alias_cost_pruned;
static u64 g_alias_cost_iterations;
static u64 g_alias_cost_hits;
static bool g_alias_cost_reported;

static void ppc_guest_alias_cost_report(void);

static bool ppc_guest_alias_cost_on(void) {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("BLUEWAKE_TRACE_ALIAS_COST") != NULL ? 1 : 0;
        if (enabled)
            atexit(ppc_guest_alias_cost_report);
    }
    return enabled != 0;
}

static void ppc_guest_alias_cost_report(void) {
    if (g_alias_cost_reported || getenv("BLUEWAKE_TRACE_ALIAS_COST") == NULL)
        return;
    g_alias_cost_reported = true;
    fprintf(stderr,
            "[alias-cost] resolve_calls=%llu pruned_by_bounds=%llu "
            "iterations=%llu hits=%llu aliases=%u overlap_mem1=%d\n",
            (unsigned long long)g_alias_cost_calls,
            (unsigned long long)g_alias_cost_pruned,
            (unsigned long long)g_alias_cost_iterations,
            (unsigned long long)g_alias_cost_hits, g_guest_alias_count,
            g_ppc_guest_aliases_overlap_mem1 ? 1 : 0);
}

static void ppc_guest_alias_recompute_bounds(void) {
    g_guest_alias_min_start = UINT32_MAX;
    g_guest_alias_max_end = 0u;
    ppc_guest_alias_set_overlap_mem1(false);
    for (u32 i = 0u; i < g_guest_alias_count; ++i) {
        const PPCGuestAlias* alias = &g_guest_aliases[i];
        if (alias->linked_start < g_guest_alias_min_start)
            g_guest_alias_min_start = alias->linked_start;
        const u64 alias_end = (u64)alias->linked_start + alias->size;
        if (alias_end > g_guest_alias_max_end)
            g_guest_alias_max_end = alias_end;
        if (alias->linked_start < GC_RAM_BASE + GC_MAIN_RAM_SIZE &&
            alias_end > GC_RAM_BASE)
            ppc_guest_alias_set_overlap_mem1(true);
    }
}

GXRUNTIME_EXPORT void ppc_guest_alias_clear(void) {
    for (u32 i = 0; i < g_guest_alias_count; ++i) {
        if (g_guest_aliases[i].owns_storage)
            free(g_guest_aliases[i].storage);
        g_guest_aliases[i].storage = NULL;
    }
    g_guest_alias_count = 0u;
    g_guest_alias_min_start = UINT32_MAX;
    g_guest_alias_max_end = 0u;
    g_guest_alias_overlapping = false;
    ppc_guest_alias_set_overlap_mem1(false);
}

static bool ppc_guest_alias_add_storage(u32 linked_start, u32 size,
                                        u8* storage, bool owns_storage) {
    if (size == 0u || storage == NULL ||
        g_guest_alias_count >= PPC_GUEST_ALIAS_MAX)
        return false;

    PPCGuestAlias* alias = &g_guest_aliases[g_guest_alias_count++];
    alias->linked_start = linked_start;
    alias->size = size;
    alias->storage = storage;
    alias->owns_storage = owns_storage;
    ppc_guest_alias_index_insert(g_guest_alias_count - 1u);
    if (alias->linked_start < g_guest_alias_min_start)
        g_guest_alias_min_start = alias->linked_start;
    const u64 alias_end = (u64)alias->linked_start + alias->size;
    if (alias_end > g_guest_alias_max_end)
        g_guest_alias_max_end = alias_end;
    if (alias->linked_start < GC_RAM_BASE + GC_MAIN_RAM_SIZE &&
        alias_end > GC_RAM_BASE)
        ppc_guest_alias_set_overlap_mem1(true);
    if (getenv("BLUEWAKE_TRACE_GUEST_ALIASES") != NULL &&
        alias->linked_start < 0x81516E20u &&
        (u64)alias->linked_start + alias->size > 0x81512AC0u) {
        fprintf(stderr,
                "[guest-alias-target] start=0x%08X size=0x%08X storage=%p\n",
                alias->linked_start, alias->size, (void*)alias->storage);
    }
    return true;
}

GXRUNTIME_EXPORT bool ppc_guest_alias_add(u32 linked_start, u32 size,
                                          const u8* initial_bytes) {
    if (size == 0u)
        return false;
    u8* storage = (u8*)calloc(1u, size);
    if (storage == NULL)
        return false;
    if (initial_bytes != NULL)
        memcpy(storage, initial_bytes, size);
    if (!ppc_guest_alias_add_storage(linked_start, size, storage, true)) {
        free(storage);
        return false;
    }
    return true;
}

GXRUNTIME_EXPORT bool ppc_guest_alias_add_shared(u32 linked_start, u32 size,
                                                 u8* storage) {
    return ppc_guest_alias_add_storage(linked_start, size, storage, false);
}

GXRUNTIME_EXPORT bool ppc_guest_alias_get_storage(u32 linked_start, u32 size,
                                                  u8** storage) {
    if (storage == NULL)
        return false;
    for (u32 i = 0u; i < g_guest_alias_count; ++i) {
        const PPCGuestAlias* alias = &g_guest_aliases[i];
        if (alias->linked_start == linked_start && alias->size == size) {
            *storage = alias->storage;
            return true;
        }
    }
    return false;
}

GXRUNTIME_EXPORT bool ppc_guest_alias_remove(u32 linked_start, u32 size) {
    for (u32 i = 0u; i < g_guest_alias_count; ++i) {
        PPCGuestAlias* alias = &g_guest_aliases[i];
        if (alias->linked_start != linked_start || alias->size != size)
            continue;
        if (alias->owns_storage)
            free(alias->storage);
        if (i + 1u < g_guest_alias_count) {
            memmove(alias, alias + 1u,
                    (g_guest_alias_count - i - 1u) * sizeof(*alias));
        }
        g_guest_alias_count--;
        g_guest_aliases[g_guest_alias_count] = (PPCGuestAlias){0};
        ppc_guest_alias_index_rebuild();
        ppc_guest_alias_recompute_bounds();
        return true;
    }
    return false;
}

GXRUNTIME_EXPORT bool ppc_guest_alias_resolve(u32 address, u32 size,
                                              u8** pointer,
                                              u32* journal_offset) {
    if (pointer == NULL || size == 0u)
        return false;
    const bool cost = ppc_guest_alias_cost_on();
    if (cost)
        g_alias_cost_calls++;
    if (address < g_guest_alias_min_start ||
        (u64)address + size > g_guest_alias_max_end) {
        if (cost)
            g_alias_cost_pruned++;
        return false;
    }
    if (!g_guest_alias_overlapping) {
        u32 lo = 0u, hi = g_guest_alias_count;
        while (lo < hi) { // last alias whose start is at or below the address
            const u32 mid = lo + (hi - lo) / 2u;
            if (cost)
                g_alias_cost_iterations++;
            if (g_guest_aliases[g_guest_alias_sorted[mid]].linked_start <= address)
                lo = mid + 1u;
            else
                hi = mid;
        }
        if (lo == 0u)
            return false;
        const PPCGuestAlias* alias = &g_guest_aliases[g_guest_alias_sorted[lo - 1u]];
        if (size > alias->size ||
            address - alias->linked_start > alias->size - size)
            return false;
        if (cost)
            g_alias_cost_hits++;
        *pointer = alias->storage + (address - alias->linked_start);
        if (journal_offset != NULL)
            *journal_offset = address - GC_RAM_BASE;
        return true;
    }
    for (u32 i = 0; i < g_guest_alias_count; ++i) {
        if (cost)
            g_alias_cost_iterations++;
        const PPCGuestAlias* alias = &g_guest_aliases[i];
        if (address < alias->linked_start || size > alias->size ||
            address - alias->linked_start > alias->size - size)
            continue;
        if (cost)
            g_alias_cost_hits++;
        *pointer = alias->storage + (address - alias->linked_start);
        if (journal_offset != NULL)
            *journal_offset = address - GC_RAM_BASE;
        return true;
    }
    return false;
}

bool cpu_init(CPUState* cpu) {
    memset(cpu, 0, sizeof(*cpu));

    cpu->ram_size = GC_MAIN_RAM_SIZE;
    cpu->ram = (u8*)calloc(1, cpu->ram_size);
    if (!cpu->ram) {
        fprintf(stderr, "error: failed to allocate %u bytes for RAM\n", cpu->ram_size);
        return false;
    }

    return true;
}

void cpu_free(CPUState* cpu) {
    if (cpu->ram) {
        free(cpu->ram);
        cpu->ram = NULL;
    }
}

void cpu_reset(CPUState* cpu) {
    u8* ram = cpu->ram;
    u32 ram_size = cpu->ram_size;
    PPCExternalRead external_read = cpu->external_read;
    PPCExternalWrite external_write = cpu->external_write;
    PPCExternalRead32 external_read32 = cpu->external_read32;
    PPCExternalWrite32 external_write32 = cpu->external_write32;
    PPCInstructionFallback instruction_fallback = cpu->instruction_fallback;
    PPCHostCall host_call = cpu->host_call;
    PPCExternalPointer external_pointer = cpu->external_pointer;
    void* external_user_data = cpu->external_user_data;
    u8* exram = cpu->exram;
    u32 exram_size = cpu->exram_size;
    PPCSPRRead spr_read = cpu->spr_read;
    PPCSPRWrite spr_write = cpu->spr_write;
    PPCCacheControl cache_control = cpu->cache_control;

    memset(cpu, 0, sizeof(*cpu));
    cpu->ram = ram;
    cpu->ram_size = ram_size;
    cpu->external_read = external_read;
    cpu->external_write = external_write;
    cpu->external_read32 = external_read32;
    cpu->external_write32 = external_write32;
    cpu->instruction_fallback = instruction_fallback;
    cpu->host_call = host_call;
    cpu->external_pointer = external_pointer;
    cpu->external_user_data = external_user_data;
    cpu->exram = exram;
    cpu->exram_size = exram_size;
    cpu->spr_read = spr_read;
    cpu->spr_write = spr_write;
    cpu->cache_control = cache_control;

    if (cpu->ram)
        memset(cpu->ram, 0, cpu->ram_size);
}

bool ppc_add_overflowed(u32 a, u32 b, u32 result) {
    return (((a ^ result) & (b ^ result)) >> 31) != 0;
}

void ppc_set_xer_ov(CPUState* cpu, bool ov) {
    cpu->xer = (cpu->xer & ~0x40000000u) | (ov ? 0x40000000u : 0u);
    if (ov)
        cpu->xer |= 0x80000000u;
}

static bool g_ppc_lazy_fp_enabled = true;

void ppc_lazy_fp_set_enabled(bool enabled) {
    g_ppc_lazy_fp_enabled = enabled;
}

bool ppc_fp_available(CPUState* cpu, u32 cia) {
    if (!g_ppc_lazy_fp_enabled || (cpu->msr & PPC_MSR_FP))
        return true;
    ppc_take_exception(cpu, PPC_EXC_FP_UNAVAILABLE, PPC_VECTOR_FP_UNAVAILABLE, cia, 0);
    return false;
}

void ppc_fallback_instruction(CPUState* cpu, u32 raw, u32 cia) {
    if (cpu->instruction_fallback) {
        cpu->instruction_fallback(cpu, raw, cia);
        return;
    }

    (void)raw;
    ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
}

bool ppc_host_call(CPUState* cpu, u32 address) {
    return cpu->host_call ? cpu->host_call(cpu, address) : false;
}

bool ppc_native_region_available(CPUState* cpu, u32 start, u32 end) {
    if (!cpu || !cpu->host_call)
        return true;

    u32 saved_addr = cpu->external_addr;
    u32 saved_value = cpu->external_value;
    u8 saved_rid = cpu->external_rid;
    cpu->external_addr = start;
    cpu->external_value = end;
    cpu->external_rid = PPC_NATIVE_REGION_QUERY_PENDING;
    bool blocked = cpu->host_call(cpu, PPC_HOST_CALL_NATIVE_REGION_QUERY);
    bool handled = cpu->external_rid == PPC_NATIVE_REGION_QUERY_HANDLED;
    cpu->external_addr = saved_addr;
    cpu->external_value = saved_value;
    cpu->external_rid = saved_rid;
    return handled && !blocked;
}

void ppc_system_call_exception(CPUState* cpu, u32 cia) {
    ppc_take_exception(cpu, PPC_EXC_SYSTEM_CALL, PPC_VECTOR_SYSTEM_CALL, cia + 4u, 0);
}

void ppc_dsi_exception(CPUState* cpu, u32 ea, u32 cia, u32 dsisr) {
    cpu->dar = ea;
    cpu->dsisr = dsisr;
    ppc_take_exception(cpu, PPC_EXC_DSI, PPC_VECTOR_DSI, cia, 0);
}

void ppc_alignment_exception(CPUState* cpu, u32 ea, u32 cia) {
    cpu->dar = ea;
    ppc_take_exception(cpu, PPC_EXC_ALIGNMENT, PPC_VECTOR_ALIGNMENT, cia, 0);
}

u32 ppc_mftb(CPUState* cpu, u16 tbr, u32 cia) {
    if ((tbr == 268 || tbr == 269) && cpu->spr_read)
        return cpu->spr_read(cpu, tbr, cia);
    if (tbr == 268)
        return (u32)cpu->timebase;
    if (tbr == 269)
        return (u32)(cpu->timebase >> 32);

    ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
    return 0;
}
static s32 gqr_scale(u32 value) {
    return sign_extend(value & 0x3Fu, 6);
}


static u32 psq_type_size(u8 type) {
    switch (type) {
    case 0: return 4;
    case 4:
    case 6: return 1;
    case 5:
    case 7: return 2;
    default: return 0;
    }
}

/* Quantized load/store semantics mirror Dolphin's interpreter (the chassis
 * lockstep oracle) exactly:
 *  - psq_l/psq_st (non-indexed) require only HID2.LSQE; the indexed forms
 *    are never gated. PSE is not checked, and no alignment exceptions are
 *    raised (unaligned accesses go straight to memory).
 *  - Invalid GQR types 1-3 load 0.0 into both lanes and store nothing.
 *  - Dequantization: f32(int) * f32 power-of-two, rounded once to f32.
 *  - Quantization: round the lane to f32 first, multiply by the f32
 *    power-of-two scale, clamp in f32, truncate. NaN quantizes to 0
 *    (matching SType(NaN-after-clamp) in release Dolphin on arm64). */
static f64 psq_dequant(f64 value, s32 scale) {
    if (scale == 0)
        return (f64)(f32)value;
    return (f64)(f32)ldexp(value, -scale);
}

static f64 psq_load_value(CPUState* cpu, u32 ea, u8 type, s32 scale) {
    switch (type) {
    case 0:
        return f64_value(convert_to_double(mem_read32(cpu, ea)));
    case 4:
        return psq_dequant((f64)mem_read8(cpu, ea), scale);
    case 5:
        return psq_dequant((f64)mem_read16(cpu, ea), scale);
    case 6:
        return psq_dequant((f64)(s8)mem_read8(cpu, ea), scale);
    case 7:
        return psq_dequant((f64)(s16)mem_read16(cpu, ea), scale);
    default:
        return 0.0;
    }
}

static s64 psq_quantize_int(f64 value, s64 min_value, s64 max_value, s32 scale) {
    f32 conv = (f32)value * ldexpf(1.0f, scale);
    if (isnan(conv))
        return 0;
    if (conv <= (f32)min_value)
        return min_value;
    if (conv >= (f32)max_value)
        return max_value;
    return (s64)conv;
}

static void psq_store_value(CPUState* cpu, u32 ea, u8 type, s32 scale, f64 value) {
    switch (type) {
    case 0:
        mem_write32(cpu, ea, convert_to_single_ftz(f64_bits(value)));
        break;
    case 4:
        mem_write8(cpu, ea, (u8)psq_quantize_int(value, 0, 255, scale));
        break;
    case 5:
        mem_write16(cpu, ea, (u16)psq_quantize_int(value, 0, 65535, scale));
        break;
    case 6:
        mem_write8(cpu, ea, (u8)(s8)psq_quantize_int(value, -128, 127, scale));
        break;
    case 7:
        mem_write16(cpu, ea, (u16)(s16)psq_quantize_int(value, -32768, 32767, scale));
        break;
    }
}

static bool psq_check_enabled(CPUState* cpu, bool indexed, u32 cia) {
    if (!indexed && (cpu->hid2 & PPC_HID2_LSQE) == 0) {
        ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
        return false;
    }
    return true;
}

bool ppc_psq_load(CPUState* cpu, u8 frD, u32 ea, bool w, u8 gqr_index, bool indexed, u32 cia) {
    if (!psq_check_enabled(cpu, indexed, cia))
        return false;

    u32 gqr = cpu->gqr[gqr_index & 7u];
    s32 scale = gqr_scale(gqr >> 24);
    u8 type = (u8)((gqr >> 16) & 7u);
    u32 size = psq_type_size(type);
    if (size == 0) { /* invalid GQR type: both lanes read as 0.0 */
        cpu->fpr[frD] = 0.0;
        cpu->ps1[frD] = 0.0;
        return true;
    }

    cpu->fpr[frD] = psq_load_value(cpu, ea, type, scale);
    cpu->ps1[frD] = w ? 1.0 : psq_load_value(cpu, ea + size, type, scale);
    return true;
}

bool ppc_psq_store(CPUState* cpu, u8 frS, u32 ea, bool w, u8 gqr_index, bool indexed, u32 cia) {
    if (!psq_check_enabled(cpu, indexed, cia))
        return false;

    u32 gqr = cpu->gqr[gqr_index & 7u];
    s32 scale = gqr_scale(gqr >> 8);
    u8 type = (u8)(gqr & 7u);
    u32 size = psq_type_size(type);
    if (size == 0) /* invalid GQR type: nothing is stored */
        return true;

    psq_store_value(cpu, ea, type, scale, cpu->fpr[frS]);
    if (!w)
        psq_store_value(cpu, ea + size, type, scale, cpu->ps1[frS]);
    return true;
}

void ppc_rfi(CPUState* cpu, u32 cia) {
    if (cpu->msr & PPC_MSR_PR) {
        ppc_program_exception(cpu, PPC_PROGRAM_PRIV, cia);
        return;
    }

    cpu->msr = (cpu->msr & ~PPC_MSR_RFI_MASK) | (cpu->srr1 & PPC_MSR_RFI_MASK);
    cpu->msr &= ~PPC_MSR_POW;
    cpu->pc = cpu->srr0 & ~3u;
}

void ppc_dcbz_l(CPUState* cpu, u32 ea, u32 cia) {
    if (cpu->msr & PPC_MSR_PR) {
        ppc_program_exception(cpu, PPC_PROGRAM_PRIV, cia);
        return;
    }

    if ((cpu->hid2 & PPC_HID2_LCE) == 0) {
        ppc_program_exception(cpu, PPC_PROGRAM_ILLEGAL, cia);
        return;
    }

    u32 block = ea & ~31u;
    u32 slot = (block >> 5) & 511u;
    bool hit = cpu->locked_cache_valid[slot] && cpu->locked_cache_tag[slot] == block;
    bool first_hit_error = hit && (cpu->hid2 & PPC_HID2_DCHERR) == 0;

    if (hit) {
        cpu->hid2 |= PPC_HID2_DCHERR;
        if (first_hit_error && (cpu->hid2 & PPC_HID2_DCHEE) &&
            (cpu->msr & PPC_MSR_EE) && (cpu->msr & PPC_MSR_ME)) {
            ppc_take_exception(cpu, PPC_EXC_MACHINE_CHECK, PPC_VECTOR_MACHINE_CHECK,
                               cia, PPC_SRR1_MACHINE_CHECK_DCBZL);
        }
    } else {
        cpu->locked_cache_valid[slot] = true;
        cpu->locked_cache_tag[slot] = block;
    }

    for (u32 i = 0; i < 32; i += 4)
        mem_write32(cpu, block + i, 0);
}

u32 ppc_eciwx(CPUState* cpu, u32 ea, u32 cia) {
    if ((cpu->ear & PPC_EAR_ENABLE) == 0) {
        ppc_dsi_exception(cpu, ea, cia, PPC_DSI_EAR_DISABLED);
        return 0;
    }

    if ((ea & 3u) != 0) {
        ppc_alignment_exception(cpu, ea, cia);
        return 0;
    }

    u8 rid = (u8)(cpu->ear & 0xFu);
    cpu->external_addr = ea;
    cpu->external_rid = rid;
    cpu->external_read_count++;
    if (cpu->external_read32)
        return cpu->external_read32(cpu, ea, rid);
    return 0;
}

void ppc_ecowx(CPUState* cpu, u32 ea, u32 value, u32 cia) {
    if ((cpu->ear & PPC_EAR_ENABLE) == 0) {
        ppc_dsi_exception(cpu, ea, cia, PPC_DSI_EAR_DISABLED);
        return;
    }

    if ((ea & 3u) != 0) {
        ppc_alignment_exception(cpu, ea, cia);
        return;
    }

    u8 rid = (u8)(cpu->ear & 0xFu);
    cpu->external_addr = ea;
    cpu->external_value = value;
    cpu->external_rid = rid;
    cpu->external_write_count++;
    if (cpu->external_write32)
        cpu->external_write32(cpu, ea, value, rid);
}

void ppc_tlbie(CPUState* cpu, u32 ea, u32 cia) {
    if (cpu->msr & PPC_MSR_PR) {
        ppc_program_exception(cpu, PPC_PROGRAM_PRIV, cia);
        return;
    }

    cpu->tlb_last_vps = (ea >> 12) & 0xFFFFu;
    cpu->tlb_last_index = (ea >> 12) & 0x3Fu;
    cpu->tlb_invalidate_count++;
}

bool ppc_trap_condition(u8 to, u32 a, u32 b) {
    s32 sa = (s32)a;
    s32 sb = (s32)b;

    return ((sa < sb) && (to & 0x10u)) ||
           ((sa > sb) && (to & 0x08u)) ||
           ((sa == sb) && (to & 0x04u)) ||
           ((a < b) && (to & 0x02u)) ||
           ((a > b) && (to & 0x01u));
}
