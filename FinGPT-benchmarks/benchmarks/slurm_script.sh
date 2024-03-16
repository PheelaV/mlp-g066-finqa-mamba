#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:2
#SBATCH --job-name=fin_gpt_benchmarks
#SBATCH --time=0-02:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

# mkdir -p /disk/scratch/${STUDENT_ID}


# export TMPDIR=/disk/scratch/${STUDENT_ID}/
# export TMP=/disk/scratch/${STUDENT_ID}/

# mkdir -p ${TMP}/datasets/
# export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mamba
python benchmarks.py --dataset fpb,fiqa,tfns,nwgi --base_model pythia --batch_size 16 --max_length 128 --from_remote True