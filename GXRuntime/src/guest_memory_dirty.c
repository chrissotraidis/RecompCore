// SPDX-License-Identifier: GPL-3.0-or-later
#include "gxruntime/guest_memory_dirty.h"

#include <string.h>

#define DOL_MEM1_TRACKED_SIZE 0x02000000u
#define DOL_MEM1_DIRTY_PAGE_SHIFT 12u
#define DOL_MEM1_DIRTY_PAGE_SIZE (1u << DOL_MEM1_DIRTY_PAGE_SHIFT)
#define DOL_MEM1_DIRTY_PAGE_COUNT \
    (DOL_MEM1_TRACKED_SIZE / DOL_MEM1_DIRTY_PAGE_SIZE)

static u64 g_page_epochs[DOL_MEM1_DIRTY_PAGE_COUNT];
static u64 g_next_epoch;

static bool physical_range(u32 address, u32 size, u32* start, u32* end) {
    if (size == 0u)
        return false;
    const u32 physical = address & 0x3FFFFFFFu;
    const u64 physical_end = (u64)physical + size;
    if (physical >= DOL_MEM1_TRACKED_SIZE ||
        physical_end > DOL_MEM1_TRACKED_SIZE)
        return false;
    *start = physical >> DOL_MEM1_DIRTY_PAGE_SHIFT;
    *end = (u32)(physical_end - 1u) >> DOL_MEM1_DIRTY_PAGE_SHIFT;
    return true;
}

void dol_guest_memory_dirty_reset(void) {
    memset(g_page_epochs, 0, sizeof(g_page_epochs));
    g_next_epoch = 0u;
}

void dol_guest_memory_dirty_mark(u32 address, u32 size) {
    u32 start;
    u32 end;
    if (!physical_range(address, size, &start, &end))
        return;
    const u64 epoch = ++g_next_epoch;
    for (u32 page = start; page <= end; ++page)
        g_page_epochs[page] = epoch;
}

bool dol_guest_memory_dirty_epoch(u32 address, u32 size, u64* epoch) {
    u32 start;
    u32 end;
    if (epoch == NULL || !physical_range(address, size, &start, &end))
        return false;
    u64 newest = 0u;
    for (u32 page = start; page <= end; ++page) {
        if (g_page_epochs[page] > newest)
            newest = g_page_epochs[page];
    }
    *epoch = newest;
    return true;
}
