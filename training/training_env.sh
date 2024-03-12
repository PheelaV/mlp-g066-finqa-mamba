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
conda install -y \
    pytest pandas numpy tqdm ipykernel ipywidgets packaging nbconvert \
    pytorch=2.2 torchvision torchaudio pytorch-cuda=12.1 
    -c pytorch \
    -c nvidia
# arm mac:
# conda install -y pytorch::pytorch torchvision torchaudio -c pytorch

# the usual
conda install -y pytest pandas numpy tqdm ipykernel ipywidgets packaging nbconvert

# for mamba 
# You need to install transformers from main until transformers=4.39.0 is released.
# this might give an import error in which case just into 
# ~/miniconda3/envs/mlp_g066_training/lib/python3.11/site-packages/trl/core.py
# and comment out the import statement that is being reported "something top k blah blah"
# https://huggingface.co/state-spaces/mamba-2.8b-hf 
pip install git+https://github.com/huggingface/transformers@main

pip install -U datasets trl wandb peft
pip install causal-conv1d>=1.2.0
pip install mamba-ssm

conda list
echo "-----------------------"
echo "DONE"


# for the distributed stuff
# pip install deepspeed bitsandbytes

# mamba is a bit tricky to install
# if pytorch is too old or too new
# https://github.com/state-spaces/mamba/issues/217

# torch and packaging is required by the conv_1d but you can't install those at once becase of the way
# mamba-ssm and conv_1d packages are set up.

