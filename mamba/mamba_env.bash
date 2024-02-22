#!/bin/bash
set -e
CONDA_ENV_NAME=mamba
PYTHON_VER=3.10

if conda env list | grep $CONDA_ENV_NAME ; then
    conda env remove -n $CONDA_ENV_NAME
fi

eval "$(conda shell.bash hook)"
conda create -n $CONDA_ENV_NAME -y python=$PYTHON_VER 
conda activate $CONDA_ENV_NAME
conda install -y numpy gensim pandas matplotlib pytest scipy pillow tqdm pytorch::pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# pip3 install --upgrade pip
pip3 install packaging
pip3 install causal-conv1d>=1.1.0
pip3 install mamba-ssm

conda list
echo "-----------------------"
echo "DONE"
