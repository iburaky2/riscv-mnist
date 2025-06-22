# riscv-mnist
This is an educational project made to learn about the RISC-V ISA and Verilog-AMS.
It consists of a RISC-V CPU programmed to classify digits from the MNIST dataset.
A linear classifier written in C used for the firmware.
The training is done in Python.
Images are read from an analog camera sensor connected to an ADC.
The CPU accesses the camera by reading from a specific memory address, and a memory mux is used for address mapping.

## Setup
Install the [RISC-V GNU Compiler Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain) for compiling the firmware.<br>
If on Ubuntu:
```
sudo apt install python3 gcc-riscv64-unknown-elf
```

Install [PyTorch](https://pytorch.org/get-started/locally/) for training the model.
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Download the MNIST database, train the model, and compile the firmware:
```
./setup.sh
```

## Simulation
Simulation is performed using Cadence Xcelium.
```
./run.sh
```
## RISC-V CPU
This project uses the [PicoRV32 RISC-V CPU](https://github.com/YosysHQ/picorv32).
