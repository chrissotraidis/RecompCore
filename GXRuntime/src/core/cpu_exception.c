#include "core/cpu.h"
#include <stdio.h>
#include <stdlib.h>

static u32 exception_vector_address(u32 msr, u32 vector) {
    return ((msr & PPC_MSR_IP) ? 0xFFF00000u : 0u) + vector;
}

static u32 exception_msr(u32 old_msr, u32 exception) {
    u32 clear = PPC_MSR_POW | PPC_MSR_EE | PPC_MSR_PR | PPC_MSR_FP |
                PPC_MSR_FE0 | PPC_MSR_SE | PPC_MSR_BE | PPC_MSR_FE1 |
                PPC_MSR_IR | PPC_MSR_DR | PPC_MSR_PM | PPC_MSR_RI |
                PPC_MSR_LE;
    if (exception & PPC_EXC_MACHINE_CHECK)
        clear |= PPC_MSR_ME;

    u32 next = old_msr & ~clear;
    if (old_msr & PPC_MSR_ILE)
        next |= PPC_MSR_LE;
    return next;
}

void ppc_take_exception(CPUState* cpu, u32 exception, u32 vector, u32 srr0, u32 srr1_info) {
    u32 old_msr = cpu->msr;
    cpu->srr0 = srr0;
    cpu->srr1 = (old_msr & PPC_MSR_RFI_MASK) | srr1_info;
    cpu->exception |= exception;
    cpu->msr = exception_msr(old_msr, exception);
    if (getenv("BLUEWAKE_TRACE_AUDIO_OBJECT") != NULL &&
        srr0 >= 0x80270000u && srr0 < 0x802A0000u) {
        fprintf(stderr,
                "[cpu-exception] cia=0x%08X exception=0x%08X vector=0x%08X "
                "old_msr=0x%08X new_msr=0x%08X srr1=0x%08X\n",
                srr0, exception, vector, old_msr, cpu->msr, cpu->srr1);
    }
    cpu->pc = exception_vector_address(cpu->msr, vector);
}

void ppc_program_exception(CPUState* cpu, u32 cause, u32 cia) {
    cpu->program_exception |= cause;
    ppc_take_exception(cpu, PPC_EXC_PROGRAM, PPC_VECTOR_PROGRAM, cia, cause);
}
