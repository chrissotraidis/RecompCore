// SPDX-License-Identifier: GPL-3.0-or-later
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include "core/cpu.h"

u8 bw_test_mem1[4096];
static u8 exram[4096], aliased[16];
int main(void) {
    CPUState cpu = {0};
    cpu.ram = bw_test_mem1; cpu.ram_size = sizeof bw_test_mem1;
    cpu.exram = exram; cpu.exram_size = sizeof exram;
    ppc_guest_alias_clear();
    u32 offset = 123;
    assert(get_ram_ptr(&cpu, GC_RAM_BASE, 4, &offset) == bw_test_mem1 && offset == 0);
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 4092, 4, NULL) == bw_test_mem1 + 4092);
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 4093, 4, NULL) == NULL);
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 4096, 0, NULL) == bw_test_mem1 + 4096);
    assert(get_ram_ptr(&cpu, 0x90000000u + 4092, 4, NULL) == exram + 4092);
    assert(get_ram_ptr(&cpu, 0x90000000u + 4093, 4, NULL) == NULL);
    assert(get_ram_ptr(&cpu, 0xC0000000u + 4092, 4, NULL) == bw_test_mem1 + 4092);
    const u32 starts[] = {GC_RAM_BASE, GC_RAM_BASE + 4, GC_RAM_BASE - 4,
                         0xC0000000u, 0x90000000u, 0xD0000000u};
    for (unsigned i = 0; i < sizeof starts / sizeof starts[0]; ++i) {
        assert(get_ram_ptr(&cpu, starts[i], 4097, NULL) == NULL);
        assert(get_ram_ptr(&cpu, starts[i], UINT32_MAX, NULL) == NULL);
    }
    assert(ppc_guest_alias_add_shared(GC_RAM_BASE + 256, sizeof aliased, aliased));
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 256, sizeof aliased, NULL) == aliased);
    assert(get_ram_ptr(&cpu, 0xC0000000u + 256, sizeof aliased, NULL) == aliased);
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 256, UINT32_MAX, NULL) == NULL);
    ppc_guest_alias_clear();
    cpu.exram_size = cpu.ram_size = 0;
#ifndef BW_GUEST_MEM1
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 4, 1, NULL) == NULL);
#endif
    assert(get_ram_ptr(&cpu, 0x90000000u + 4, 1, NULL) == NULL);
    return 0;
}
