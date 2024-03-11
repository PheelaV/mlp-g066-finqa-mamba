# from datasets import load_dataset

# from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM

# from custom_sft_trainer import SFTTrainer
from custom_training import CustomSFTTrainer as SFTTrainer
import custom_training
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

import utils
from collections import namedtuple

import wandb

from datasets import load_dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str)
parser.add_argument("--lora", type=bool, default=False)
parser.add_argument("--model_type", type=str)
parser.add_argument("--test_feature", type=bool, default=False)
args = parser.parse_args()
if args.model_type == "pythia":
    target_modules = ["query_key_value"]
    if args.model_size == "l":
        model_id = "EleutherAI/pythia-2.8b-deduped"
    elif args.model_size == "m":
        model_id = "EleutherAI/pythia-1.4b-deduped"
    elif args.model_size == "s":
        model_id = "EleutherAI/pythia-70m-deduped"
elif args.model_type == "mamba":
    target_modules = ["x_proj", "embeddings", "in_proj", "out_proj"]
    if args.model_size == "l":
        model_id = "state-spaces/mamba-2.8b-hf"  # OOM
    elif args.model_size == "m":
        model_id = "state-spaces/mamba-1.4b-hf"
    elif args.model_size == "s":
        model_id = "state-spaces/mamba-130m-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_id)
# dataset = load_dataset("Abirate/english_quotes", split="train")
# dataset = load_dataset("Abirate/english_quotes", split="train[:10%]")
model.config.use_cache = (
    False  # https://github.com/huggingface/transformers/issues/29505
)

dataset_args = namedtuple(
    "args",
    [
        "dataset",
        "max_length",
        "from_remote_data",
        "test_dataset",
        "instruct_template",
        "num_workers",
        "output_dir",
    ],
)
dataset_args = dataset_args(
    "sentiment-train,headline,finred*3,ner*15", 512, False, None, "default", None, "./"
)
# dataset_args = dataset_args("convfinqa", 512, False, None, "default", None)

dataset = utils.get_dataset(args=dataset_args, tokenizer=tokenizer, return_text=False)

del tokenizer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-3,
    evaluation_strategy="steps",
    report_to="wandb",
    eval_steps=0.1,
    remove_unused_columns=False,  # important because we are injecting custom metadata for the loss function
)
# lora_config =  LoraConfig(
#     r=8,
#     target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
#     task_type="CAUSAL_LM",
#     bias="none"
# )
lora_config = LoraConfig(
    r=8,
    target_modules=target_modules,
    task_type="CAUSAL_LM",
    bias="none",
)

complete_args = {}
complete_args["prompt_loss_weight"] = 0.1

# instruction_template = "Instruction: :"
# response_template = "Answer: "
# collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     peft_config=lora_config,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     max_seq_length=512,
#     dataset_text_field="input_ids",
#     # data_collator=collator
#     data_collator=custom_training.CustomDataCollatorSeq2Seq(tokenizer, padding=True),
#     # tokenized_datasets=True,
#     # prompt_loss_weight=complete_args["prompt_loss_weight"]
# )
trainer = SFTTrainer(
    model=model,
    # tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config if args.lora else None,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_seq_length=512,
    # dataset_text_field="input_ids",
    # data_collator=collator
    data_collator=custom_training.CustomDataCollatorSeq2Seq(
        tokenizer, padding=True, prompt_loss_weight=complete_args["prompt_loss_weight"], test_feature=args.test_feature
    ),
    tokenized_datasets=True,
    prompt_loss_weight=complete_args["prompt_loss_weight"],
    test_feature=args.test_feature
)
trainer.train()
