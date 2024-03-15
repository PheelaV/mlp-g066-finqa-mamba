#!/bin/bash

# this should pass in order for us to proceed
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 64 \
--max_length 512 \
--base_model ../../training/finetuned_models/test_models/mamba_s

python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 64 \
--max_length 512 \
--base_model ../../training/finetuned_models/test_models/pythia_s
