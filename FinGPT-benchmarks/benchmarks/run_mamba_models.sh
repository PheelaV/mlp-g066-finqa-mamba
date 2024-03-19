#!/bin/bash

# Directory containing all the mamba models
models_dir="../../training/test_models/mamba"

# Commands to run python benchmarks.py with specific base_model paths

## BIG

# # mamba-big_mamba_l_mt_0_2024_03_16_1625
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 32 \
# --max_length 512 \
# --base_model "${models_dir}/mamba-big_mamba_l_mt_0_2024_03_16_1625/"

## MEDIUM

# # mamba-medium_mamba_m_mt_0_2024_03_15_1748
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 32 \
# --max_length 512 \
# --base_model "${models_dir}/mamba-medium_mamba_m_mt_0_2024_03_15_1748/final-best/"

# # mamba-medium_mamba_m_mt_1_2024_03_16_0538
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 32 \
# --max_length 512 \
# --base_model "${models_dir}/mamba-medium_mamba_m_mt_1_2024_03_16_0538/"

# # mamba-medium_mamba_m_mt_2_2024_03_16_0258
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 32 \
# --max_length 512 \
# --base_model "${models_dir}/mamba-medium_mamba_m_mt_2_2024_03_16_0258/"

# mamba-medium_mamba_m_mqsq_1_b_2024_03_17_0501_mamba_m_mqsq_1_b_mt_0_2024_03_17_1458
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/mamba-medium_mamba_m_mqsq_1_b_2024_03_17_0501_mamba_m_mqsq_1_b_mt_0_2024_03_17_1458/"

# mamba-medium_mamba_m_mqsq_1_b_2024_03_17_0501_mamba_m_mqsq_1_b_mt_2_2024_03_17_1722
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/mamba-medium_mamba_m_mqsq_1_b_2024_03_17_0501_mamba_m_mqsq_1_b_mt_2_2024_03_17_1722/final-best/"


## SMALL

# mamba-small_mamba_s_mt_0_2024_03_14_0457
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 32 \
--max_length 512 \
--base_model "${models_dir}/mamba-small_mamba_s_mt_0_2024_03_14_0457/"

# mamba-small_mamba_s_mt+_0_2024_03_16_0615
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 32 \
--max_length 512 \
--base_model "${models_dir}/mamba-small_mamba_s_mt+_0_2024_03_16_0615/final-best/"

# mamba-small_mamba_s_mt+_1_2024_03_16_2005
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 32 \
--max_length 512 \
--base_model "${models_dir}/mamba-small_mamba_s_mt+_1_2024_03_16_2005/final-best/"

# mamba-small_mamba_s_mt+_2_2024_03_16_0616
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 32 \
--max_length 512 \
--base_model "${models_dir}/mamba-small_mamba_s_mt+_2_2024_03_16_0616/final-best/"

# mamba-small_mamba_s_mt_1_2024_03_14_0457
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 32 \
--max_length 512 \
--base_model "${models_dir}/mamba-small_mamba_s_mt_1_2024_03_14_0457/"

# mamba-small_mamba_s_mt_2_2024_03_14_0943
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 32 \
--max_length 512 \
--base_model "${models_dir}/mamba-small_mamba_s_mt_2_2024_03_14_0943/"

# mamba_s_mt_0_2024_03_13_0356
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 32 \
--max_length 512 \
--base_model "${models_dir}/mamba_s_mt_0_2024_03_13_0356/checkpoint-26271/"
