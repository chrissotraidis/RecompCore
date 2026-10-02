// SPDX-License-Identifier: GPL-3.0-or-later
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include "core/cpu.h"

u8 bw_test_mem1[0x02000000u];
static u8 exram[4096], aliased[16], upper_alias[16];
static u32 journal_count, journal_offset, journal_size, io_count, io_addr, io_size;
static u64 io_value;
static void journal(u32 offset, u32 size, void* user) {
    assert(user == &journal_count);
    journal_count++; journal_offset = offset; journal_size = size;
}
static u64 io_read(CPUState* cpu, u32 addr, u8 size) {
    (void)cpu; io_count++; io_addr = addr; io_size = size;
    return 0x123456789ABCDEF0ull;
}
static void io_write(CPUState* cpu, u32 addr, u64 value, u8 size) {
    (void)cpu; io_count++; io_addr = addr; io_size = size; io_value = value;
}
int main(void) {
    CPUState cpu = {0};
    cpu.ram = bw_test_mem1; cpu.ram_size = sizeof bw_test_mem1;
    cpu.exram = exram; cpu.exram_size = sizeof exram;
    ppc_guest_alias_clear();
    u32 offset = 123;
    assert(get_ram_ptr(&cpu, GC_RAM_BASE, 4, &offset) == bw_test_mem1 && offset == 0);
    const u32 ram_size = sizeof bw_test_mem1;
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + ram_size - 4, 4, NULL) == bw_test_mem1 + ram_size - 4);
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + ram_size - 3, 4, NULL) == NULL);
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + ram_size, 0, NULL) == bw_test_mem1 + ram_size);
    assert(get_ram_ptr(&cpu, 0x90000000u + 4092, 4, NULL) == exram + 4092);
    assert(get_ram_ptr(&cpu, 0x90000000u + 4093, 4, NULL) == NULL);
    assert(get_ram_ptr(&cpu, 0xC0000000u + 4092, 4, NULL) == bw_test_mem1 + 4092);
    const u32 starts[] = {GC_RAM_BASE, GC_RAM_BASE + 4, GC_RAM_BASE - 4,
                         0xC0000000u, 0x90000000u, 0xD0000000u};
    for (unsigned i = 0; i < sizeof starts / sizeof starts[0]; ++i) {
        assert(get_ram_ptr(&cpu, starts[i], ram_size + 1, NULL) == NULL);
        assert(get_ram_ptr(&cpu, starts[i], UINT32_MAX, NULL) == NULL);
    }
    assert(ppc_guest_alias_add_shared(GC_RAM_BASE + 256, sizeof aliased, aliased));
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 256, sizeof aliased, NULL) == aliased);
    assert(get_ram_ptr(&cpu, 0xC0000000u + 256, sizeof aliased, NULL) == aliased);
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 256, UINT32_MAX, NULL) == NULL);
    /* BlueWake expands MEM1 to 32 MiB for linked REL data. An alias there
     * must bypass direct RAM both when inserted and after registry changes. */
    assert(ppc_guest_alias_add_shared(0x81F00000u, sizeof upper_alias, upper_alias));
    assert(ppc_guest_alias_remove(GC_RAM_BASE + 256, sizeof aliased));
    assert(g_ppc_guest_aliases_overlap_mem1);
    assert(!ppc_dispatch_poll_read_stable(&cpu, 0x81F00000u, 4));
    mem_write32(&cpu, 0x81F00000u, 0xABCDEF12u);
    assert(upper_alias[0] == 0xAB && upper_alias[3] == 0x12);
    assert(mem_read32(&cpu, 0xC1F00000u) == 0xABCDEF12u);
    assert(bw_test_mem1[0x01F00000u] == 0);
    ppc_guest_alias_clear();
    assert(!g_ppc_guest_aliases_overlap_mem1);
    assert(ppc_dispatch_poll_read_stable(&cpu, 0x81F00000u, 4));
    assert(ppc_guest_alias_add_shared(0x81F00000u, sizeof upper_alias, upper_alias));
    assert(g_ppc_guest_aliases_overlap_mem1);
    assert(get_ram_ptr(&cpu, 0x81F00000u, 4, NULL) == upper_alias);
    assert(ppc_guest_alias_remove(0x81F00000u, sizeof upper_alias));
    assert(!g_ppc_guest_aliases_overlap_mem1);

    g_mem_write_journal = journal; g_mem_write_journal_user = &journal_count;
    cpu.reserve_valid = true; cpu.reserve_addr = GC_RAM_BASE + 64;
    mem_write64(&cpu, 0xC0000048u, 0x0123456789ABCDEFull);
    assert(!cpu.reserve_valid && journal_count == 1 && journal_offset == 72 && journal_size == 8);
    assert(mem_read64(&cpu, GC_RAM_BASE + 72) == 0x0123456789ABCDEFull);
    assert(bw_test_mem1[72] == 0x01 && bw_test_mem1[79] == 0xEF);
    cpu.reserve_valid = true;
    mem_write32(&cpu, GC_RAM_BASE + 128, 0x12345678u);
    mem_write16(&cpu, GC_RAM_BASE + 132, 0xABCDu);
    mem_write8(&cpu, GC_RAM_BASE + 134, 0xEFu);
    assert(cpu.reserve_valid && journal_count == 4 && journal_offset == 134 && journal_size == 1);
    assert(mem_read32(&cpu, 0xC0000080u) == 0x12345678u);
    assert(mem_read16(&cpu, 0xC0000084u) == 0xABCDu && mem_read8(&cpu, 0xC0000086u) == 0xEFu);
    mem_write32(&cpu, 0x90000000u, 0x11223344u);
    assert(journal_count == 4 && mem_read32(&cpu, 0xD0000000u) == 0x11223344u);
    cpu.external_read = io_read; cpu.external_write = io_write;
    assert(mem_read32(&cpu, 0xCC000020u) == 0x9ABCDEF0u);
    assert(io_count == 1 && io_addr == 0xCC000020u && io_size == 4);
    mem_write16(&cpu, 0xCC000022u, 0x4321u);
    assert(io_count == 2 && io_addr == 0xCC000022u && io_size == 2 && io_value == 0x4321u);
    mem_write32(&cpu, GC_RAM_BASE + ram_size - 1, 0x55667788u);
    assert(io_count == 3 && io_addr == GC_RAM_BASE + ram_size - 1 && io_size == 4);
    assert(journal_count == 4 && bw_test_mem1[ram_size - 1] == 0);
    g_mem_write_journal = NULL; g_mem_write_journal_user = NULL;
    cpu.exram_size = cpu.ram_size = 0;
#ifndef BW_GUEST_MEM1
    assert(get_ram_ptr(&cpu, GC_RAM_BASE + 4, 1, NULL) == NULL);
#endif
    assert(get_ram_ptr(&cpu, 0x90000000u + 4, 1, NULL) == NULL);
    return 0;
}
