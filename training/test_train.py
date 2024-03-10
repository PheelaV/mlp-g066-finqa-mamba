from datasets import load_dataset
# from trl import SFTTrainer
# from custom_sft_trainer import SFTTrainer
from custom_training import CustomSFTTrainer as SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments


model_id = "state-spaces/mamba-1.4b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
dataset = load_dataset("Abirate/english_quotes", split="train[:10%]")
training_args = TrainingArguments(
    output_dir="./results",
    # num_train_epochs=3,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="quote",
)
trainer.train()
