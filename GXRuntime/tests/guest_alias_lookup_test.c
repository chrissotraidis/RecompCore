// SPDX-License-Identifier: GPL-3.0-or-later
// ppc_guest_alias_resolve against the rule it implements: the first alias, in
// registration order, that holds the whole access. The resolver's shortcuts
// (the alias found last, the page index, the sorted search) must give that
// answer for every alias set and access, including after removes.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "core/cpu.h"

u8 bw_test_mem1[0x02000000u];

typedef struct { u32 start, size; u8* storage; } Alias;
static Alias model[4096];
static u32 model_count;
static u8 pool[1]; // storage identities only: the resolver never reads them
static u64 rng = 0x9E3779B97F4A7C15ull;
static u64 queries;

static u32 next(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (u32)rng;
}

static u8* storage_for(u32 index) { return pool + (uintptr_t)index * 64u; }

static bool reference(u32 address, u32 size, u8** pointer) {
    for (u32 i = 0; i < model_count; ++i) {
        const Alias* a = &model[i];
        if (address >= a->start && size <= a->size && address - a->start <= a->size - size) {
            *pointer = a->storage + (address - a->start);
            return true;
        }
    }
    return false;
}

static void check(u32 address, u32 size) {
    u8* want = NULL;
    u8* got = (u8*)1;
    u32 offset = 0xFFFFFFFFu;
    const bool expected = reference(address, size, &want);
    const bool found = ppc_guest_alias_resolve(address, size, &got, &offset);
    queries++;
    if (found != expected || (found && (got != want || offset != address - GC_RAM_BASE))) {
        fprintf(stderr, "mismatch at 0x%08X size %u: got %d %p, want %d %p\n",
                address, size, found, (void*)got, expected, (void*)want);
        assert(false);
    }
}

static void add(u32 start, u32 size) {
    assert(model_count < 4096u);
    model[model_count] = (Alias){start, size, storage_for(model_count)};
    assert(ppc_guest_alias_add_shared(start, size, model[model_count].storage));
    model_count++;
}

static void remove_at(u32 i) {
    assert(ppc_guest_alias_remove(model[i].start, model[i].size));
    for (u32 j = i; j + 1u < model_count; ++j)
        model[j] = model[j + 1u];
    model_count--;
}

static void reset(void) {
    ppc_guest_alias_clear();
    model_count = 0u;
}

// Edges of every alias, then random accesses across the span, each size twice
// in a row (the second is answered by the alias found last when it holds it).
static void probe(u32 low, u32 high, u32 random_count) {
    static const u32 sizes[] = {1u, 2u, 4u, 8u};
    for (u32 i = 0; i < model_count; ++i) {
        const u32 s = model[i].start, end = model[i].start + model[i].size;
        const u32 edges[] = {s - 1u, s, s + 1u, end - 8u, end - 4u, end - 1u, end, end + 1u,
                             (s & ~0xFFFu) - 1u, s & ~0xFFFu, (end + 0xFFFu) & ~0xFFFu};
        for (u32 e = 0; e < sizeof edges / sizeof edges[0]; ++e)
            for (u32 z = 0; z < 4u; ++z) {
                check(edges[e], sizes[z]);
                check(edges[e], sizes[z]);
            }
    }
    for (u32 n = 0; n < random_count; ++n) {
        const u32 address = low + next() % (high - low);
        const u32 size = sizes[next() & 3u];
        check(address, size);
        check(address, size);
    }
}

// Non-overlapping aliases laid out like Wind Waker's linked REL data: packed
// small ones (several per page) and a few large ones, with gaps.
static void layout(u32 base, u32 count, u32 max_size) {
    u32 at = base;
    for (u32 i = 0; i < count; ++i) {
        at += next() % 3u == 0u ? next() % 0x3000u : next() % 64u;
        const u32 size = next() % 8u == 0u ? 1u + next() % max_size : 1u + next() % 256u;
        add(at, size);
        at += size;
    }
}

int main(void) {
    // About 1,900 aliases in the uncached window, as the game registers them.
    reset();
    layout(0xC0400FD0u, 1900u, 0x0000C000u);
    probe(0xC0400000u, model[model_count - 1u].start + model[model_count - 1u].size + 0x2000u, 400000u);

    // Removes and re-adds keep the answers (the registry moves; the alias found
    // last may then be another one).
    for (u32 round = 0; round < 50u; ++round) {
        const u32 i = next() % model_count;
        const Alias gone = model[i];
        remove_at(i);
        probe(gone.start - 0x100u, gone.start + gone.size + 0x100u, 200u);
        add(gone.start, gone.size);
    }
    probe(0xC0400000u, 0xC2000000u, 100000u);

    // The registry's capacity, small aliases packed into a few pages.
    reset();
    for (u32 i = 0; i < 4096u; ++i)
        add(0x81820000u + i * 24u, 1u + next() % 24u);
    probe(0x8181F000u, 0x81820000u + 4096u * 24u + 0x1000u, 200000u);

    // One alias; then one per page boundary.
    reset();
    add(0x81F00000u, 16u);
    probe(0x81EFF000u, 0x81F02000u, 10000u);
    reset();
    for (u32 i = 0; i < 64u; ++i)
        add(0xC1000000u + i * 0x1000u - 4u, 8u);
    probe(0xC0FFF000u, 0xC1042000u, 50000u);

    // A span wider than the page table: the sorted search answers.
    reset();
    add(0x80001000u, 64u);
    add(0xC1F00000u, 64u);
    layout(0xC0000000u, 200u, 0x1000u);
    probe(0x80000000u, 0xC2000000u, 100000u);

    // Overlapping aliases: registration order decides (the shortcuts are off).
    reset();
    layout(0xC0400000u, 300u, 0x4000u);
    for (u32 i = 0; i < 100u; ++i) {
        const Alias* a = &model[next() % model_count];
        add(a->start + a->size / 2u, a->size + 32u);
    }
    probe(0xC0400000u, model[299].start + model[299].size + 0x1000u, 200000u);

    reset();
    check(0xC0400000u, 4u);
    printf("guest alias lookup: %llu accesses matched the reference\n", (unsigned long long)queries);
    return 0;
}
