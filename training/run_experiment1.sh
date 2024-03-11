#!/bin/bash

python train.py --config config_mt.json \
    --base_model pythia-small \
    --run_name pythia_mt_ \
    --prompt_loss_weight 0.1