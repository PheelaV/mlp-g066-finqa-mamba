import lm_eval
from lm_eval.logging_utils import WandbLogger
import os
import wandb

models_dir="../../training/finetuned_models"

model_dir="/mamba-medium_mamba_m_mt_2_2024_03_16_0258"

wandb.init(
    project="mlp-g066-benchmarks",
    name="mamba_m_mt_2",
)

results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={models_dir}{model_dir},trust_remote_code=True",
    # tasks="arc_challenge,arc_easy,lambada,hallaswag,piqa,winogrande",
    # tasks="hellaswag",
    tasks="piqa",
    log_samples=True,
)

wandb_logger = WandbLogger(
    project="lm-eval-harness-integration", job_type="eval"
)  # or empty if wandb.init(...) already called before
wandb_logger.post_init(results)
wandb_logger.log_eval_result()
wandb_logger.log_eval_samples(results["samples"])
print(results)
