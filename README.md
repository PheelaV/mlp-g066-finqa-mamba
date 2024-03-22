# Financial Q&A Using Mamba/Pythia
This report explores the application of the Mamba model, a novel state space model (SSM), in the domain of financial question answering (Q&A), challenging the dominance of transformer-based architectures such as FinBERT, FinGPT, and BloombergGPT. Overall, the results show that fine-tuning Mamba with the typical financial datasets, did not outperform the traditional transformer architectures. Despite facing challenges, initial findings indicate areas for improvement and further research.

## Training
The training has been done through training/train.py (which is formed from the Jupyter Notebook train.ipynb). To run, the training environment must be setup with training_env.sh. 

## Evaluation
The benchmarking is done within FinGPT-benchmarks/benchmarks as well as EleutherAI/lm-evaluation-harness. There are some scripts to run both Mamba and Pythia.
Much of this code has been taken from the open-source project [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) and modified after the fact to suit our needs. The main file is benchmarks2.py which enriches the benchmarking suite with the help of [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
