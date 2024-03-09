#!/bin/bash
set -e
CONDA_ENV_NAME=mlp_g066_training
PYTHON_VER=3.11

if conda env list | grep $CONDA_ENV_NAME ; then
    conda env remove -n $CONDA_ENV_NAME
fi

eval "$(conda shell.bash hook)"
conda create -n $CONDA_ENV_NAME -y python=$PYTHON_VER 
conda activate $CONDA_ENV_NAME
# linux:
# conda install -y \
#     pytorch==2.1.1 \
#     torchvision==0.16.1 \
#     torchaudio==2.1.1 \
#     pytorch-cuda=12.1 \
#     -c pytorch -c nvidia

# arm mac:
conda install -y pytorch::pytorch torchvision torchaudio -c pytorch

# the usual
conda install -y transformers datasets pytest pandas numpy tqdm ipykernel ipywidgets packaging

pip install -U datasets trl wandb

conda list
echo "-----------------------"
echo "DONE"
