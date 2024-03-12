#!/usr/bin/env python
# coding: utf-8

# # Train LLMs
# optionally LoRA or distributed

# this notebook produces `train.py`, just execute `jupyter nbconvert --to script 'train.ipynb'`
# 
# ## Configuration
# - configuration will be taken from the defaults
# - defaults can be overwritten by a supplied json config
# - everythin can be overwritten by cli arguments
# 
# 
# **defaults < config < cli**
# 
# huggingface will cache datasets and their transformations
# sometimes this might come in handy:
# `dataset.cleanup_cache_files()` 
# 
# This script will first check if the tokenized dataset is present in `data/` and use that, otherwise it will interface with huggingface and attempt to create a new one. This check is done by `dataset_id` which consists of the name of the particular tokenizer and the maximum sequence length (as the ones over length are ignored).

# In[1]:


import os
import json

import argparse
from datetime import datetime

import torch

import utils
import custom_training

from transformers import AutoModelForCausalLM


def is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def load_config(json_filepath):
    if os.path.isfile(json_filepath):
        try:
            with open(json_filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(
                f"Error loading configuration file {json_filepath}: {e}. Using default command line arguments."
            )
    else:
        print(
            f"Configuration file {json_filepath} not found. Using default command line arguments."
        )
    return {}


# In[ ]:


# import logging
# import socket
# from datetime import datetime, timedelta

# import torch

# from torch.autograd.profiler import record_function
# from torchvision import models

# logging.basicConfig(
#    format="%(levelname)s:%(asctime)s %(message)s",
#    level=logging.INFO,
#    datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger: logging.Logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)

# TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# def trace_handler(prof: torch.profiler.profile):
#    # Prefix for file names.
#    host_name = socket.gethostname()
#    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
#    file_prefix = f"{host_name}_{timestamp}"

#    # Construct the trace file.
#    prof.export_chrome_trace(f"{file_prefix}.json.gz")

#    # Construct the memory timeline file.
#    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


# In[3]:


def main(args):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    data_dir_path = os.path.join(args.working_dir, "data")
    model_dir_path = os.path.join(args.working_dir, "finetuned_models")
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    if args.local_rank == 0:
        print(
            f'Script is running in {"interactive" if IS_INTERACTIVE else "non-interactive"} mode'
        )

    if IS_INTERACTIVE:
        from tqdm.notebook import tqdm

        os.environ["WANDB_NOTEBOOK_NAME"] = "train.ipynb"
        from importlib import reload

        reload(custom_training)
        reload(utils)

    # Parse the model name and determine if it should be fetched from a remote source
    model_name = utils.parse_model_name(args)
    if args.local_rank == 0:
        print(f"Using model: {model_name}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = utils.get_tokenizer(args, model_name)
    dataset = utils.get_dataset(args, tokenizer)
    if args.local_rank == 0:
        print(f"Dataset loaded: {dataset}")

    if args.seed_data:
        exit()
    
    if "mamba" in args.base_model:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # https://github.com/huggingface/transformers/issues/29505
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # load_in_8bit=True,
            # device_map="auto",
            trust_remote_code=True,
        )

    # Print model architecture for the first process in distributed training
    if args.local_rank == 0:
        print(model)

    # Create a timestamp for model saving
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H%M")

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = utils.get_tokenizer(args, model_name)

   
    trainer, training_args, common_args = utils.get_trainer(
        args, model, tokenizer, dataset, formatted_time
    )

    # Clear CUDA cache and start training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    import wandb

    wandb.init(
        project="mlp-g066-mamba",
        name=args.run_name,
        config=common_args,
        dir=args.working_dir,
    )

    if args.local_rank == 0:
        start_time = datetime.now()
        print(f"Start Time: {start_time.isoformat()}")
        wandb.log({"start_time":start_time})
        
    trainer.train()
    
    if args.local_rank == 0:
        end_time = datetime.now()
        wandb.log({"end_time":end_time})
        print(f"End Time: {end_time.isoformat()}")
    # Save the fine-tuned model
    trainer.save_model(os.path.join(args.working_dir, "finetuned_models", args.base_model))


# In[4]:


if __name__ == "__main__":
    global IS_INTERACTIVE
    IS_INTERACTIVE = is_interactive()
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        type=str,
        help="Optional path to JSON configuration file",
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--lora_r", default=0, type=int, help="Lora rank, 0 for no lora"
    )
    parser.add_argument("--run_name", default="local-test", type=str)
    parser.add_argument("--dataset", default="convfinqa", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument(
        "--base_model",
        required=False,
        default="pythia-small",
        type=str,
        # choices=["mamba-small", "pythia-small", "mamba-medium", "pythia-small", "mamba-big", "pythia-big"],
    )
    parser.add_argument("--max_length", default=512, type=int)
    # parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument(
        "--batch_size", default=4, type=int, help="The train batch size per device"
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="The learning rate"
    )
    parser.add_argument("--num_epochs", default=8, type=int, help="The training epochs")
    parser.add_argument(
        "--gradient_steps",
        default=8,
        type=float,
        help="The gradient accumulation steps",
    )
    parser.add_argument(
        "--num_workers",
        default="8",
        type=str,
        help="Dataloader workers - number or 'all' to use all available cores",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.05,
        type=float,
        help="Ration of steps used for learning rate warmup "
        "(gradually increases LR before the normal schedule takes over)",
    )
    parser.add_argument(
        "--ds_config",
        default="./ds_config.json",
        type=str,
        help="Deeppspeed configuration file",
    )
    parser.add_argument("--scheduler", default="linear", type=str)
    parser.add_argument(
        "--working_dir",
        default="./",
        type=str,
        help="Location where the model and logs will be saved as well as datasets read from.",
    )
    parser.add_argument("--instruct_template", default="default")
    parser.add_argument("--load_best_model", default="False", type=bool)
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--evaluation_strategy", default="steps", type=str)
    parser.add_argument("--eval_steps", default=0.1, type=float)
    parser.add_argument("--eval_accumulation_steps", default=None, type=int)
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "paged_adamw_8bit"],
    )
    parser.add_argument(
        "--from_remote_data",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Fetch the dataset form hugging face",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Resume training from a checkpoint of a previously saved model in the working_dir/finetuned_models",
    )
    parser.add_argument(
        "--model_from_local",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Get a local model from the working_dir/finetuned_models",
    )
    parser.add_argument(
        "--prompt_loss_weight", default=1.0, type=float, help="Prompt loss weight"
    )
    parser.add_argument(
        "--distributed",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable per device batch and gradient accumulation",
    )
    parser.add_argument(
        "--fp16", default=False, type=bool, help="Enable fp16 precision"
    )
    parser.add_argument(
        "--bf16", default=False, type=bool, help="Enable bf16 precision"
    )
    parser.add_argument(
        "--seed_data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dry run for seeding the data - say your compute node is slow or does"
        " not have access to internet and you need to seed the data locally.",
    )
    parser.add_argument(
        # currently there seems to be a bug which causes OOM on too large of an evaluation set
        # this is a problem with the HF trainer
        "--eval",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Evaluate trhoughout training",
    )

    global args  # Namespace args
    # Parse the command line arguments and update defaults with JSON configuration
    # config_args, remaining_argv = (
    #     parser.parse_known_args("") if is_interactive() else parser.parse_known_args()
    # )
    config_args = parser.parse_args("") if is_interactive() else parser.parse_args()

    json_config_path = (
        config_args.config
    )  # Use the --config command line argument to specify JSON config file
    config_defaults = load_config(json_config_path) if json_config_path else {}
    if "_comment" in config_defaults:
        config_defaults.pop("_comment")
    # Update parser defaults based on JSON configuration
    # parser.set_defaults(**config_defaults)
    # Now parse the rest of the arguments with the updated defaults
    # args = parser.parse_args(remaining_argv)
    args = parser.parse_args("") if is_interactive() else parser.parse_args()

    if args.num_workers == "all":
        args.num_workers = os.cpu_count()
    elif args.num_workers.isdigit():
        args.num_workers = int(args.num_workers)
    else:
        raise ValueError("num_workers must be 'all' or an integer")

    main(args)

