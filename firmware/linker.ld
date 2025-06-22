OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
  .init 0x00000000 : {
    *(.init*)
  }

  .text : {
    *(.text*)
  }

  .rodata : {
    *(.rodata*)
  }

  .data : {
    *(.data*)
  }

  .bss : {
    *(.bss*)
    *(COMMON)
  }

  .stack  0x00010000 : {
    _stack_top = .;
    . = . + 0x400;
    _stack_bottom = .;
  }
}
