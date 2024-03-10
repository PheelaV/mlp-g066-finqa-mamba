from datasets import load_dataset

# from trl import SFTTrainer
# from custom_sft_trainer import SFTTrainer
from custom_training import CustomSFTTrainer as SFTTrainer
import custom_training
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

import utils
from collections import namedtuple
import multiprocessing


# model_id = "state-spaces/mamba-2.8b-hf" # OOM
# model_id = "state-spaces/mamba-1.4b-hf"
model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
# dataset = load_dataset("Abirate/english_quotes", split="train[:10%]")

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

dataset_args.dataset = "convfinqa"
dataset_args.max_length = 512
dataset_args.from_remote_data = False
dataset_args.test_dataset = None
dataset_args.instruct_template = "default"
dataset_args.num_workers = multiprocessing.cpu_count()
dataset = utils.get_dataset(args=dataset_args, tokenizer=tokenizer)["train"]

training_args = TrainingArguments(
    output_dir="./results",
    # num_train_epochs=3,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-3,
)
lora_config = LoraConfig(
    r=8,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    # dataset_text_field="quote",
    data_collator=custom_training.CustomDataCollatorSeq2Seq(tokenizer, padding=True),
    tokenized_datasets=True,
)
trainer.train()
