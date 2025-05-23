#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_LAUNCH_BLOCKING=1 python3 newflclient.py $1 $2 > client$1$2 2>&1 &; disown $!