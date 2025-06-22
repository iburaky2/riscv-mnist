#!/bin/bash
# Move to a temporary folder to avoid cluttering the root directory with Xcelium run files.
mkdir -p tmp
cd tmp
    # -gui \
    # -access +rwc \
xrun -ams \
    ../logic/camera_sensor.vams \
    ../logic/adc.vams \
    ../logic/camera.vams \
    ../logic/picorv32.v \
    ../logic/memory_mux.v \
    ../tb/testbench.vams \
    -top testbench \
    -analogcontrol ../test/test.scs
