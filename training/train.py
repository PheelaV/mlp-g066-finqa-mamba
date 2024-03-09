# %% [markdown]
# DISCLAIMER: a lot of this code is take/modified from FinGPT

# %%
import os

import argparse
from datetime import datetime
from functools import partial

import torch 
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq
    )
import datasets

import utils
import custom_training

def is_interactive():
    import __main__ as main
    return not hasattr(main, "__file__")

# %%
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
    dataset_list = utils.load_dataset(args.dataset, args.from_remote)
    dataset_train = datasets.concatenate_datasets([d['train'] for d in dataset_list]).shuffle(seed=42)
    if args.test_dataset:
        dataset_list = utils.load_dataset(args.test_dataset, args.from_remote)
    dataset_test = datasets.concatenate_datasets([d['test'] for d in dataset_list])
    dataset = datasets.DatasetDict({'train': dataset_train, 'test': dataset_test})
    # Display first sample from the training dataset
    print(dataset['train'][0])
    # Filter out samples that exceed the maximum token length and remove unused columns
    dataset = dataset.map(partial(utils.tokenize, args, tokenizer, prompt_in_label=True))
    print('original dataset length: ', len(dataset['train']))
    dataset = dataset.filter(lambda x: not x['exceed_max_length'])
    print('filtered dataset length: ', len(dataset['train']))
    dataset = dataset.remove_columns(['instruction', 'input', 'output', 'exceed_max_length'])
    return dataset

def get_trainer(args, model, tokenizer, dataset, formatted_time):
    """
        Create the trainer and training arguments
    """
    if args.distributed:
        training_args = TrainingArguments(
            output_dir=f"finetuned_models/{args.run_name}_{formatted_time}",
            logging_steps=args.log_interval,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_steps,
            dataloader_num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.scheduler,
            save_steps=args.eval_steps,
            eval_steps=args.eval_steps,
            fp16=True,
            # fp16_full_eval=True,
            deepspeed=args.ds_config,
            evaluation_strategy=args.evaluation_strategy,
            load_best_model_at_end=args.load_best_model,
            remove_unused_columns=False,
            report_to="wandb",
            run_name=args.run_name,
        )
        trainer = custom_training.CustomSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=custom_training.CustomDataCollatorSeq2Seq(tokenizer, padding=True),
        )
    else:
        training_args = TrainingArguments(
            output_dir=f"finetuned_models/{args.run_name}_{formatted_time}",
            logging_steps=args.log_interval,
            num_train_epochs=args.num_epochs,
            # per_device_train_batch_size=args.batch_size,
            # per_device_eval_batch_size=args.batch_size,
            # gradient_accumulation_steps=args.gradient_steps,
            dataloader_num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.scheduler,
            save_steps=args.eval_steps,
            eval_steps=args.eval_steps,
            # fp16=True,
            # fp16_full_eval=True,
            # deepspeed=args.ds_config,
            evaluation_strategy=args.evaluation_strategy,
            load_best_model_at_end=args.load_best_model,
            remove_unused_columns=False,
            report_to='wandb',
            run_name=args.run_name,
        )
        trainer = custom_training.CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=custom_training.CustomDataCollatorSeq2Seq(tokenizer, padding=True),
        )
    return trainer, training_args


# %%
def main(args):
    print(f"Script is running in {'interactive' if IS_INTERACTIVE else 'non-interactive'} mode")

    if IS_INTERACTIVE:
        from tqdm.notebook import tqdm
        os.environ['WANDB_NOTEBOOK_NAME'] = "train_local_test"
        from importlib import reload
        reload(custom_training)
        reload(utils)


    # Parse the model name and determine if it should be fetched from a remote source
    model_name = utils.parse_model_name(args.base_model, args.from_remote)

    if "mamba" in args.base_model:
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # load_in_8bit=True,
            # device_map="auto",
            trust_remote_code=True
        )

    # Print model architecture for the first process in distributed training
    if args.local_rank == 0:
        print(model)

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    tokenizer = get_tokenizer(model_name)
    dataset = get_dataset(args, tokenizer)

    # Create a timestamp for model saving
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H%M")

    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    tokenizer = get_tokenizer(model_name)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model = model.to(device)
    trainer, training_args = get_trainer(args, model, tokenizer, dataset, formatted_time)

    # Clear CUDA cache and start training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(training_args.output_dir)

# %%
if __name__ == "__main__":
    global IS_INTERACTIVE 
    IS_INTERACTIVE= is_interactive()
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--run_name", default="local-test", type=str)
    parser.add_argument(
        "--dataset", required=False, default="convfinqa", type=str
    )
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
    parser.add_argument(
        "--num_epochs", default=8, type=float, help="The training epochs"
    )
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
    parser.add_argument("--from_remote", default=True, type=bool)
    parser.add_argument("--prompt_loss_weight", default=0, type=float)
    parser.add_argument("--distributed", default=False, type=bool)

    # parser.add_argument("--from_remote", default=not IS_INTERACTIVE, type=bool)
    global args
    args = parser.parse_args("") if is_interactive() else parser.parse_args()

    import wandb
    config = vars(args)
    config.pop("ds_config")
    wandb.init(project="mlp-g066-mamba", name=args.run_name, config=config)
    # Run main
    main(args)

# %% [markdown]
# TODO:

# %%
# from datasets import load_dataset
# from trl import SFTTrainer
# from peft import LoraConfig
# from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
# model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")
# dataset = load_dataset("Abirate/english_quotes", split="train")
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     logging_dir='./logs',
#     logging_steps=10,
#     learning_rate=2e-3
# )
# lora_config =  LoraConfig(
#         r=8,
#         target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
#         task_type="CAUSAL_LM",
#         bias="none"
# )
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     peft_config=lora_config,
#     train_dataset=dataset,
#     dataset_text_field="quote",
# )
# trainer.train()



