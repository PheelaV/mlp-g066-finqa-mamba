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
conda install -y packaging pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y transformers==4.35.0
pip3 install --upgrade pip
pip3 install accelerate==0.25.0 bitsandbytes==0.41.3 scipy==1.11.4
pip3 install causal-conv1d==1.0.0 mamba-ssm==1.0.1

conda list
echo "-----------------------"
echo "DONE"
