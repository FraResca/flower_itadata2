#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# CUDA_LAUNCH_BLOCKING=1 python3 flserver.py
CUDA_LAUNCH_BLOCKING=1 python3 newflserver.py $1 $2 > server$1$2 2>&1 & disown $!