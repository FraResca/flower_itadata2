#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# CUDA_LAUNCH_BLOCKING=1 python3 flserver.py
CUDA_LAUNCH_BLOCKING=1 python3 centralized_fl.py $1 > centralized$1.lore 2>&1 & disown $!