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

# In[17]:


import os
import json

import argparse
from datetime import datetime
from functools import partial

import torch
from transformers import AutoTokenizer, TrainingArguments
import datasets

import utils
import custom_training


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

if not os.path.exists("data"):
    os.makedirs("data")
    
if not os.path.exists("finetuned_models"):
    os.makedirs("finetuned_models")


# In[18]:


def get_tokenizer(args, model_name):
    """
    Load the tokenizer and set the special tokens, specific to the model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if "mamba" in args.base_model or "pythia" in args.base_model:
        # these were defaults either way
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = "<|padding|>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    return tokenizer


def get_dataset(args, tokenizer):
    """
    Load the dataset and apply tokenization
    """

    tok_cls_name = (
        tokenizer.__class__.__name__[:-4]
        if tokenizer.__class__.__name__[-4:] == "Fast"
        else tokenizer.__class__.__name__
    )

    # for persistence
    dataset_id = f"{args.dataset}_{args.max_length}_{tok_cls_name}"
    # if dataset is already tokenized, load it
    # unless we specifically want the remote version
    if (not args.from_remote_data) and os.path.exists(f"data/{dataset_id}"):
        print("Using cached dataset")
        return datasets.load_from_disk(f"data/{dataset_id}")
    else:
        print("Loading dataset from remote")

    dataset_list = utils.load_dataset(args.dataset, args.from_remote_data)
    dataset_train = datasets.concatenate_datasets(
        [d["train"] for d in dataset_list]
    ).shuffle(seed=42)
    if args.test_dataset:
        dataset_list = utils.load_dataset(args.test_dataset, args.from_remote_data)
    dataset_test = datasets.concatenate_datasets([d["test"] for d in dataset_list])
    dataset = datasets.DatasetDict({"train": dataset_train, "test": dataset_test})
    # Display first sample from the training dataset
    # print(dataset["train"][0])
    # Filter out samples that exceed the maximum token length and remove unused columns
    dataset = dataset.map(
        partial(utils.tokenize, args, tokenizer, prompt_in_label=True)
    )
    print("original dataset length: ", len(dataset["train"]))
    dataset = dataset.filter(lambda x: not x["exceed_max_length"])
    print("filtered dataset length: ", len(dataset["train"]))
    dataset = dataset.remove_columns(
        ["instruction", "input", "output", "exceed_max_length"]
    )

    dataset.save_to_disk(f"data/{dataset_id}")

    return dataset


def get_trainer(args, model, tokenizer, dataset, formatted_time):
    """
    Create the trainer and training arguments
    """

    common_args = {
        "output_dir": f"finetuned_models/{args.run_name}_{formatted_time}",
        "logging_steps": args.log_interval,
        "num_train_epochs": args.num_epochs,
        "dataloader_num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.scheduler,
        "save_steps": args.eval_steps,
        "eval_steps": args.eval_steps,
        "evaluation_strategy": args.evaluation_strategy,
        "load_best_model_at_end": args.load_best_model,
        "remove_unused_columns": False,
        "report_to": "wandb",
        "run_name": args.run_name,
        "fp16": args.fp16 & torch.cuda.is_available(),
        "logging_dir": "./logs",
        "label_names":[]
    }

    if args.distributed:
        distributed_args = {
            "deepspeed": args.ds_config,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_steps,
        }
        common_args.update(distributed_args)

    training_args = TrainingArguments(**common_args)

    if args.lora_r > 0:
        from peft import (
            LoraConfig,
            # get_peft_model,
            TaskType,
        )

        if "mamba" in args.base_model:
            peft_config = LoraConfig(
                r=args.lora_r,
                target_modules=utils.lora_module_dict[args.base_model],
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=utils.lora_module_dict[args.base_model],
                bias="none",
            )
        trainer = custom_training.CustomSFTTrainer(
            model=model,
            args=training_args,
            peft_config=peft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=custom_training.CustomDataCollatorSeq2Seq(
                tokenizer, padding=True
            ),
        )

        # update args for logging
        common_args.update(**peft_config.__dict__)
    else:
        trainer = custom_training.CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=custom_training.CustomDataCollatorSeq2Seq(
                tokenizer, padding=True
            ),
        )

        # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     peft_config=lora_config,
    #     train_dataset=dataset,
    #     dataset_text_field="quote",
    # )

    import wandb

    wandb.init(project="mlp-g066-mamba", name=args.run_name, config=common_args)

    return trainer, training_args


# In[19]:


def main(args):
    mode = "interactive" if IS_INTERACTIVE else "non-interactive"

    if args.local_rank == 0:
        print(
            f"Script is running in {mode} mode"
        )

    if IS_INTERACTIVE:
        from tqdm.notebook import tqdm

        os.environ["WANDB_NOTEBOOK_NAME"] = "train.ipynb"
        from importlib import reload

        reload(custom_training)
        reload(utils)

    # Parse the model name and determine if it should be fetched from a remote source
    model_name = utils.parse_model_name(args.base_model, args.from_remote_model)
    if args.local_rank == 0:
        print(f"Using model: {model_name}")
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = get_tokenizer(args, model_name)
    dataset = get_dataset(args, tokenizer)
    if args.local_rank == 0:
        print(f"Dataset loaded: {dataset}")

    if "mamba" in args.base_model:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        model = MambaLMHeadModel.from_pretrained(
            model_name, dtype=torch.bfloat16, device="cuda"
        )
    else:
        from transformers import AutoModelForCausalLM

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
    tokenizer = get_tokenizer(args, model_name)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    if args.local_rank == 0:
        print(f"Commencing training on device: {device}, time: {formatted_time}")
    model = model.to(device)
    trainer, training_args = get_trainer(
        args, model, tokenizer, dataset, formatted_time
    )

    # Clear CUDA cache and start training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # return
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(training_args.output_dir)


# In[20]:


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
    parser.add_argument("--local_rank", default=0, type=int ,help="Local rank for distributed training")
    parser.add_argument("--lora_r", default=0, type=int, help="Lora rank, 0 for no lora")
    parser.add_argument("--run_name", default="local-test", type=str)
    parser.add_argument("--dataset", required=False, default="convfinqa", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument(
        "--base_model",
        required=False,
        default="pythia-small",
        type=str,
        choices=["mamba-small", "pythia-small"],
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
    parser.add_argument("--num_workers", default=8, type=int, help="dataloader workers")
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--ds_config", default="./config_new.json", type=str)
    parser.add_argument("--scheduler", default="linear", type=str)
    parser.add_argument("--instruct_template", default="default")
    parser.add_argument("--evaluation_strategy", default="steps", type=str)
    parser.add_argument("--load_best_model", default="False", type=bool)
    parser.add_argument("--eval_steps", default=0.1, type=float)
    parser.add_argument(
        "--from_remote_data",
        default=0,
        type=bool,
    )
    parser.add_argument(
        "--from_remote_model",
        default=1,
        type=bool,
    )
    parser.add_argument(
        "--prompt_loss_weight", default=0, type=float, help="Prompt loss weight"
    )
    parser.add_argument(
        "--distributed", default=False, type=bool, help="Enable deepspeed"
    )
    parser.add_argument(
        "--fp16", default=False, type=bool, help="Enable fp16 precision"
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
    parser.set_defaults(**config_defaults)
    # Now parse the rest of the arguments with the updated defaults
    # args = parser.parse_args(remaining_argv)
    args = parser.parse_args("") if is_interactive() else parser.parse_args()

    # Run main
    main(args)

