# mlp-g066-finqa-mamba
Investigation of utilizing Mamba in the FinQ&A domain, staking it up against traditional transformers.

The benchmarking is done within FinGPT-benchmarks/benchmarks. There are some scripts to run both Mamba and Pythia.
Much of this code has been taken from the open-source project FinGPT. The main file is benchmarks.py.

The training has been done through training/train.py. To run, the training environment must be setup with training_env.sh. 
