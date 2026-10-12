// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef GXRUNTIME_GUEST_MEMORY_DIRTY_H
#define GXRUNTIME_GUEST_MEMORY_DIRTY_H

#include "core/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tracks GPU-visible changes to physical MEM1 at page granularity. Writers
// mark ranges when the guest flushes CPU cache lines or a device completes a
// bulk transfer. Callers outside physical MEM1 receive false and must use
// content comparison instead.
//
// Two kinds of mark: a write (a device transfer, a readback written into RAM)
// and a cache flush (the guest's dcbf/dcbst, which says the CPU may have
// written there). dol_guest_memory_dirty_epoch sees both; the write epoch only
// writes. An image the GPU itself rendered into a guest address (an EFB copy)
// has no bytes in RAM to go stale, so it follows the write epoch: a flush of a
// line in the same page must not replace it with RAM's unrelated bytes.
void dol_guest_memory_dirty_reset(void);
void dol_guest_memory_dirty_mark(u32 address, u32 size);
void dol_guest_memory_flush_mark(u32 address, u32 size);
bool dol_guest_memory_dirty_epoch(u32 address, u32 size, u64* epoch);
bool dol_guest_memory_write_epoch(u32 address, u32 size, u64* epoch);

#ifdef __cplusplus
}
#endif

#endif
