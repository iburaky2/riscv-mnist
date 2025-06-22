.PHONY: all
all: firmware.mem

firmware.elf: firmware.c
	riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostartfiles -nostdlib -nodefaultlibs -Wl,--no-relax -T linker.ld -o firmware.elf firmware.c

firmware.bin: firmware.elf
	riscv64-unknown-elf-objcopy -O binary firmware.elf firmware.bin

firmware.mem: firmware.bin
	hexdump -v -e '1/4 "%08x\n"' firmware.bin > firmware.mem

clean:
	rm -f firmware.elf firmware.bin firmware.mem
