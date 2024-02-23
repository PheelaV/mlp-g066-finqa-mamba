#!/bin/bash
set -e
CONDA_ENV_NAME=mamba_chat
PYTHON_VER=3.10

if conda env list | grep $CONDA_ENV_NAME ; then
    conda env remove -n $CONDA_ENV_NAME
fi

eval "$(conda shell.bash hook)"
conda create -n $CONDA_ENV_NAME -y python=$PYTHON_VER 
conda activate $CONDA_ENV_NAME
# this step is just what worked for me, can be optimized
conda install -y \
    pytorch==2.1.1 \
    torchvision==0.16.1 \
    torchaudio==2.1.1 \
    pytorch-cuda=12.1 \
    -c pytorch -c nvidia
pip install \
    packaging torch==2.1.0 \
    transformers==4.35.0 \
    causal-conv1d==1.0.0 \
    mamba-ssm==1.0.1 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3 \
    scipy==1.11.4

conda list
echo "-----------------------"
echo "DONE"
