#!/bin/bash

# Directory containing all the pythia models
models_dir="../../training/test_models/pythia"


## BIG

# # pythia-big_pythia_l_mt_0_2024_03_16_1053
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia-big_pythia_l_mt_0_2024_03_16_1053/checkpoint-20691/"

## MEDIUM

# # pythia-medium_pythia_m_mt+_0_2024_03_17_0446
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia-medium_pythia_m_mt+_0_2024_03_17_0446/final-best/"

# # pythia-medium_pythia_m_mt_1_2024_03_16_1225
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia-medium_pythia_m_mt_1_2024_03_16_1225/final-best/"

# # pythia-medium_pythia_m_mt_2_2024_03_16_1445
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia-medium_pythia_m_mt_2_2024_03_16_1445/final-best/"

# # pythia-medsmall_pythia_ms_mt+_0_2024_03_17_0027
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia-medsmall_pythia_ms_mt+_0_2024_03_17_0027/final-best/"

# # pythia-medsmall_pythia_ms_mt+_3_2024_03_17_0514
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia-medsmall_pythia_ms_mt+_3_2024_03_17_0514/final-best/"

# # pythia-medsmall_pythia_ms_mt_0_2024_03_14_1735
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia-medsmall_pythia_ms_mt_0_2024_03_14_1735/"

# # pythia-medsmall_pythia_ms_mt_1_2024_03_14_1529
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia-medsmall_pythia_ms_mt_1_2024_03_14_1529/"

# # pythia-medsmall_pythia_ms_mt_2_2024_03_14_1601
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia-medsmall_pythia_ms_mt_2_2024_03_14_1601/checkpoint-28469/"

# pythia_m_mt_0_2024_03_15_0539
python benchmarks.py \
--dataset fpb,fiqa,tfns,nwgi \
--force_use_model \
--batch_size 128 \
--max_length 512 \
--base_model "${models_dir}/pythia_m_mt_0_2024_03_15_0539/checkpoint-13794/"

## SMALL
#
# # pythia-small_pythia_s_mt+_0_2024_03_16_1555
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia-small_pythia_s_mt+_0_2024_03_16_1555/final-best/"
#
# # pythia-small_pythia_s_mt+_1_2024_03_16_2005
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia-small_pythia_s_mt+_1_2024_03_16_2005/final-best/"
#
# # pythia-small_pythia_s_mt+_2_2024_03_16_1555
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia-small_pythia_s_mt+_2_2024_03_16_1555/final-best/"
#
# # pythia-small_pythia_s_mt_0_2024_03_14_1430
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia-small_pythia_s_mt_0_2024_03_14_1430/"
#
# # pythia-small_pythia_s_mt_1_2024_03_14_0521
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia-small_pythia_s_mt_1_2024_03_14_0521/"
#
# # pythia-small_pythia_s_mt_1_2024_03_14_1138
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia-small_pythia_s_mt_1_2024_03_14_1138/checkpoint-28469/"
#
# # pythia-small_pythia_s_mt_2_2024_03_14_0944
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia-small_pythia_s_mt_2_2024_03_14_0944/checkpoint-28469/"
#
# # pythia_s_mt_0_2024_03_13_0356
# python benchmarks.py \
# --dataset fpb,fiqa,tfns,nwgi \
# --force_use_model \
# --batch_size 128 \
# --max_length 512 \
# --base_model "${models_dir}/pythia_s_mt_0_2024_03_13_0356/checkpoint-26271/"
