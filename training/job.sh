#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:a6000:1
#SBATCH 
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00


# first of all one needs to create the folders on scratch
# - /disk/scratch/<UUD>/data
# - /disk/scratch/<UUD>/finetuned_models

# then you need to seed the data as gpu nodes do not have full access to the internet
# for example for multi-task dataset
# model is important because of the choice of tokenizer (dataset is pre-tokenized)
# python train.py --dataset "sentiment-train,headline,finred*3,ner*15" --working_dir /disk/scratch/<UUD> --seed_data --num_workers 16 --base_model mamba-small
# python train.py --dataset "sentiment-train,headline,finred*3,ner*15" --working_dir /disk/scratch/2588483 --seed_data --num_workers 16 --base_model mamba-small --shared_dir ./
export CUDA_HOME=/opt/cuda-12.2.1/

# export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=2588483
# does not owrk for some reason
# export STUDENT_ID=$(whoami)

# export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

# export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

# mkdir -p /disk/scratch/${STUDENT_ID}
# mkdir -p /disk/scratch/${STUDENT_ID}
# mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

# mkdir -p ${TMP}/data/
# mkdir -p ${TMP}/finetuned_models/
export DATASET_DIR=${TMP}/data/

export HF_DATASETS_CACHE=$DATASET_DIR

# df--gres=gpu:a6000:2

# source /home/$STUDENT_ID/miniconda3/bin/activate mlp_g066_training

CONDA_ACTIVATE_SCRIPT="/home/$STUDENT_ID/miniconda3/bin/activate"
if [[ -f "$CONDA_ACTIVATE_SCRIPT" ]]; then
    source $CONDA_ACTIVATE_SCRIPT mlp_g066_training
else
    echo "Conda activate script not found: $CONDA_ACTIVATE_SCRIPT"
    exit 1
fi

# cd ~/repos/mlp-g066-finqa-mamba/training
# accelerate launch train.py \
#     --run_name "mamba_s_mt_0-1" \
#     --base_model mamba-small \
#     --dataset "sentiment-train,headline,finred*3,ner*15" \
#     --max_length 512 \
#     --config config_mt.json \
#     --distributed 1 \
#     --eval_accumulation_steps 8 \
#     --prompt_loss_weight 0.1 \
#     --working_dir $TMP \


python train.py --config config_test.json --shared_dir /home/$STUDENT_ID/shared >train.log 2>&1 &


# python train.py \
#     --run_name "mamba_s_mt_0-1" \
#     --base_model mamba-small \
#     --dataset "sentiment-train,headline,finred*3,ner*15" \
#     --max_length 512 \
#     --config config_mt.json \
#     --eval_accumulation_steps 8 \
#     --prompt_loss_weight 0.1 \
#     --working_dir $TMP \
#     --shared_dir /home/$STUDENT_ID/shared
#     >train.log 2>&1 &
# accelerate launch train.py \
#     --run_name "pythia_s_mt_0-1" \
#     --base_model pythia-small \
#     --prompt_loss_weight 0.1 \
#     --max_length 512 \
#     --config config_mt.json \
#     --dataset "sentiment-train,headline,finred*3,ner*15" \
#     --eval_accumulation_steps 8 \
#     --distributed 1 \
#     --working_dir $TMP \
#     >train.log 2>&1 &

# python train.py --run_name "mamba_s_mt_0-1" --base_model mamba-small --dataset "sentiment-train,headline,finred*3,ner*15" --max_length 512 --config config_mt.json --eval_accumulation_steps 8 --prompt_loss_weight 0.1 --working_dir /disk/scratch/s2588483/