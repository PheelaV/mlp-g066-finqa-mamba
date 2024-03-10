# from datasets import load_dataset

# from trl import SFTTrainer
# from custom_sft_trainer import SFTTrainer
from custom_training import CustomSFTTrainer as SFTTrainer
import custom_training
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

import utils
from collections import namedtuple

import wandb


# model_id = "state-spaces/mamba-2.8b-hf" # OOM
# model_id = "state-spaces/mamba-1.4b-hf"
model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
# dataset = load_dataset("Abirate/english_quotes", split="train[:10%]")
model.config.use_cache = False # https://github.com/huggingface/transformers/issues/29505
dataset_args = namedtuple(
    "args",
    [
        "dataset",
        "max_length",
        "from_remote_data",
        "test_dataset",
        "instruct_template",
        "num_workers",
    ],
)


dataset_args = dataset_args("convfinqa", 512, False, None, "default", None)

dataset = utils.get_dataset(args=dataset_args, tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-3,
    evaluation_strategy="steps",
    report_to="wandb",
    eval_steps=10
)
lora_config = LoraConfig(
    r=8,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)

complete_args = {}
complete_args["prompt_loss_weight"] = 0.1
run1 = wandb.init(project = "wood_runs", name = "test_run1", config = complete_args)
# run1.log({'loss':loss1})
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=custom_training.CustomDataCollatorSeq2Seq(tokenizer, padding=True),
    tokenized_datasets=True,
    prompt_loss_weight=complete_args["prompt_loss_weight"]
)
trainer.train()
run1.finish()

run2 = wandb.init(project = "wood_runs", name = "test_run2", config = complete_args)
complete_args["prompt_loss_weight"] = 1.0
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=custom_training.CustomDataCollatorSeq2Seq(tokenizer, padding=True),
    tokenized_datasets=True,
    prompt_loss_weight=1.0
)
trainer.train()
run2.finish()