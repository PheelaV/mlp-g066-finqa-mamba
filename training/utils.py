# DISCLAIMER: a lot of this code is take/modified from FinGPT
import os
import datasets
import custom_training
from transformers import AutoTokenizer, TrainingArguments
from functools import partial
import torch
import json

# A dictionary to store various prompt templates.
template_dict = {"default": "Instruction: {instruction}\nInput: {input}\nAnswer: "}

# A dictionary to store the LoRA module mapping for different models.
lora_module_dict = {
    # 'chatglm2': ['query_key_value'],
    # 'falcon': ['query_key_value'],
    # 'bloom': ['query_key_value'],
    # 'internlm': ['q_proj', 'k_proj', 'v_proj'],
    # 'llama2': ['q_proj', 'k_proj', 'v_proj'],
    # 'llama2-13b': ['q_proj', 'k_proj', 'v_proj'],
    # 'llama2-13b-nr': ['q_proj', 'k_proj', 'v_proj'],
    # 'qwen': ["c_attn"],ythi
    # 'mpt': ['Wqkv'],
    # 'baichuan': ['q_proj', 'k_proj', 'v_proj'],
    "pythia": ["query_key_value"],
    "mamba": ["x_proj", "embeddings", "in_proj", "out_proj"],
}


def parse_model_name(args):
    """
    Parse the model name and return the appropriate path based on whether
    the model is to be fetched from a remote source or from a local source.

    Args:
      - args (Namespace)
        - name (str): Name of the model.
        - from_remote (bool): If True, return the remote path, else return the local path.

    Returns:
    - str: The appropriate path for the given model name.
    """
    model_paths = {
        # 'chatglm2': ('THUDM/chatglm2-6b', 'base_models/chatglm2-6b'),
        # 'llama2': ('meta-llama/Llama-2-7b-hf', 'base_models/Llama-2-7b-hf'),
        # 'llama2-13b': ('meta-llama/Llama-2-13b-hf', 'base_models/Llama-2-13b-hf'),
        # 'llama2-13b-nr': ('NousResearch/Llama-2-13b-hf', 'base_models/Llama-2-13b-hf'),
        # 'falcon': ('tiiuae/falcon-7b', 'base_models/falcon-7b'),
        # 'internlm': ('internlm/internlm-7b', 'base_models/internlm-7b'),
        # 'qwen': ('Qwen/Qwen-7B', 'base_models/Qwen-7B'),
        # 'baichuan': ('baichuan-inc/Baichuan2-7B-Base', 'base_models/Baichuan2-7B-Base'),
        # 'mpt': ('cekal/mpt-7b-peft-compatible', 'base_models/mpt-7b-peft-compatible'),
        # 'bloom': ('bigscience/bloom-7b1', 'base_models/bloom-7b1'),
        "mamba-small": "state-spaces/mamba-130m-hf",
        "pythia-small": "EleutherAI/pythia-70m-deduped",
        "mamba-medium": "state-spaces/mamba-1.4b-hf",
        "pythia-medium": "EleutherAI/pythia-1.4b-deduped",
        "mamba-big": "state-spaces/mamba-2.8b-hf",
        "pythia-big": "EleutherAI/pythia-2.8b-deduped",
    }

    if args.base_model in model_paths and not args.model_from_local:
        return model_paths[args.base_model]
    if not args.model_from_local:
        print("Didn't fina remote model, Trying to get a local model.")

    model_path = os.path.join(args.working_dir, "finetuned_models", args.base_model)
    if os.path.exists(model_path):
        return model_path
    else:
        raise ValueError(
            f"Undefined base model '{args.base_model}'. Valid model names are: {json.dumps(model_paths, indent=2)}"
        )


def get_prompt(template, instruction, input_text):
    """
    Generates a prompt based on a predefined template, instruction, and input.

    Args:
    template (str): The key to select the prompt template from the predefined dictionary.
    instruction (str): The instruction text to be included in the prompt.
    input_text (str): The input text to be included in the prompt.

    Returns:
    str: The generated prompt.

    Raises:
    KeyError: If the provided template key is not found in the template dictionary.
    """
    if not instruction:
        return input_text

    if template not in template_dict:
        raise KeyError(
            f"Template '{template}' not found. Available templates: {', '.join(template_dict.keys())}"
        )

    return template_dict[template].format(instruction=instruction, input=input_text)


def test_mapping(args, feature):
    """
    Generate a mapping for testing purposes by constructing a prompt based on given instructions and input.

    Args:
    args (Namespace): A namespace object that holds various configurations, including the instruction template.
    feature (dict): A dictionary containing 'instruction' and 'input' fields used to construct the prompt.

    Returns:
    dict: A dictionary containing the generated prompt.

    Raises:
    ValueError: If 'instruction' or 'input' are not provided in the feature dictionary.
    """
    # Ensure 'instruction' and 'input' are present in the feature dictionary.
    if "instruction" not in feature or "input" not in feature:
        raise ValueError(
            "Both 'instruction' and 'input' need to be provided in the feature dictionary."
        )

    # Construct the prompt using the provided instruction and input.
    prompt = get_prompt(
        args.instruct_template, feature["instruction"], feature["input"]
    )

    return {
        "prompt": prompt,
    }


def tokenize(args, tokenizer, feature, prompt_in_label=False, return_text=False):
    """
    Tokenizes the input prompt and target/output for model training or evaluation.

    Args:
    args (Namespace): A namespace object containing various settings and configurations.
    tokenizer (Tokenizer): A tokenizer object used to convert text into tokens.
    feature (dict): A dictionary containing 'input', 'instruction', and 'output' fields.

    Returns:
    dict: A dictionary containing tokenized 'input_ids', 'labels', and a flag 'exceed_max_length'.
    """
    # Generate the prompt.
    prompt = get_prompt(
        args.instruct_template, feature["instruction"], feature["input"]
    )
    # Tokenize the prompt.
    prompt_ids = tokenizer(
        prompt, padding=False, max_length=args.max_length, truncation=True
    )["input_ids"]
    prompt_lens = len(prompt_ids)

    # Tokenize the target/output.
    target_ids = tokenizer(
        feature["output"].strip(),
        padding=False,
        max_length=args.max_length,
        truncation=True,
        add_special_tokens=False,
    )["input_ids"]

    # Combine tokenized prompt and target output.
    input_ids = prompt_ids + target_ids

    # Check if the combined length exceeds the maximum allowed length.
    exceed_max_length = len(input_ids) >= args.max_length

    # this is to investigate a memory leak, just desperately trying stuff
    # by replacing with reference code
    if return_text:
        return {
            "input_ids": prompt,
            "labels": prompt,
            "exceed_max_length": exceed_max_length,
            "prompt_lens": prompt_lens,
        }

    # Add an end-of-sequence (EOS) token if it's not already present
    # and if the sequence length is within the limit.
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)

    if not prompt_in_label:
        # Create label IDs for training.
        # The labels should start from where the prompt ends, and be padded for the prompt portion.
        label_ids = [tokenizer.pad_token_id] * prompt_lens + input_ids[prompt_lens:]
    else:
        label_ids = input_ids

    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length,
        "prompt_lens": prompt_lens,
    }


def load_dataset(args, names, from_remote=False, dataset_identifier_map=None):
    """
    Load one or multiple datasets based on the provided names and source location.

    Args:
    names (str): A comma-separated list of dataset names. Each name can be followed by '*n' to indicate replication.
    from_remote (bool): If True, load the dataset from Hugging Face's model hub. Otherwise, load it from a local disk.
    dataset_identifier_map (dict): A mapping from user-friendly dataset names to their actual identifiers. Used when `from_remote` is True.
    
    Returns:
    List[Dataset]: A list of loaded datasets. Each dataset is possibly replicated based on the input names.
    """

    if dataset_identifier_map is None:
        # A default dictionary to store various datasets identifiers (can be overrident)
        dataset_identifier_map = {
            "fiqa_qa": "FinGPT/fingpt-fiqa_qa",
            "convfinqa": "FinGPT/fingpt-convfinqa",
            "finsquad": "rajpurkar/squad_v2",
            "mathqa" : "microsoft/orca-math-word-problems-200k"
            # we can add more
        }

    # relevant topics for finsquad
    relevant_topics_finsquad = ['Sony_Music_Entertainment'.lower(), 'Universal_Studios', 'Royal_Dutch_Shell'.lower(), 'European_Central_Bank'.lower(), 'Financial_crisis_of_2007%E2%80%9308'.lower()]

    # Split the dataset names by commas for handling multiple datasets
    dataset_names = names.split(",")
    dataset_list = []

    for name in dataset_names:
        # Initialize replication factor to 1
        replication_factor = 1
        dataset_name = name

        # Check if the dataset name includes a replication factor
        if "*" in name:
            dataset_name, replication_factor = name.split("*")
            replication_factor = int(replication_factor)
            if replication_factor < 1:
                raise ValueError("Replication factor must be a positive integer.")
        
        # Determin that identifier is correct
        if from_remote:
            try:
                dataset_path_or_name = dataset_identifier_map[dataset_name]
            except KeyError:
                raise ValueError(f"Unknown dataset name: {dataset_name}. Please check the dataset identifier map.")
        else:
            dataset_path_or_name = os.path.join(args.working_dir, "data", dataset_name)

        if not from_remote and not os.path.exists(dataset_path_or_name):
            print(
                f"The dataset path {dataset_path_or_name} does not exist, trying remote."
            )
            try:
                dataset_path_or_name = dataset_identifier_map[dataset_name]
                from_remote = True
            except KeyError:
                raise ValueError(f"Unknown dataset name: {dataset_name}. Please check the dataset identifier map or provide a correct name.")
        print(f"Loading dataset: {dataset_path_or_name}")

        # Load the dataset
        try:
            tmp_dataset = (
                datasets.load_dataset(dataset_path_or_name)
                if from_remote
                else datasets.load_from_disk(dataset_path_or_name)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load the dataset: {str(e)}")

        #filter finsquad to have only relevant topics
        if dataset_name == 'finsquad':
            tmp_dataset = tmp_dataset.filter(lambda example: any(topic.lower() in example['title'].lower() for topic in relevant_topics_finsquad))
        
        # Check for 'test' split and create it from 'train' if necessary
        if "test" not in tmp_dataset:
            if "train" in tmp_dataset:
                tmp_dataset = tmp_dataset["train"]
                tmp_dataset = tmp_dataset.train_test_split(
                    test_size=0.2, shuffle=True, seed=42
                )
            else:
                raise ValueError("The dataset must contain a 'train' or 'test' split.")

        # Append the possibly replicated dataset to the list
        dataset_list.extend([tmp_dataset] * replication_factor)

    return dataset_list


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


def get_dataset(args, tokenizer, return_text=False):
    """
    Load the dataset and apply tokenization

    Args:
    - args (Namespace): A namespace object containing various settings and
        configurations.
        - dataset (str): The name of the dataset to be loaded.
        - max_length (int): The maximum token length allowed for the input and
            output sequences.
        - from_remote_data (bool): If True, load the dataset from Hugging Face's
            model hub. Otherwise, load it from a local disk.
        - test_dataset (str): The name of the test dataset to be loaded,
            optional.
        - instruct_template (str): The key to select the prompt template from
            the predefined dictionary.
        - num_workers (int): The number of workers to use for parallel
            processing, optional.
    - tokenizer (Tokenizer): A tokenizer object used to convert text into
        tokens.
    - return_text (bool): If True, return the dataset as text, otherwise tokenized.)
    """
    tok_cls_name = (
        tokenizer.__class__.__name__[:-4]
        if tokenizer.__class__.__name__[-4:] == "Fast"
        else tokenizer.__class__.__name__
    )

    # for persistence
    dataset_name = args.dataset.replace(",", "_").replace("*", "")

    dataset_id = f"{dataset_name}_{args.max_length}_{tok_cls_name}"
    print(dataset_id)
    # if dataset is already tokenized, load it
    # unless we specifically want the remote version
    cache_dataset_path = os.path.join(args.working_dir, "data", dataset_id)
    if (
        os.path.exists(cache_dataset_path)
        and not return_text
        and not args.from_remote_data
    ):
        print("Using cached dataset")
        return datasets.load_from_disk(cache_dataset_path)
    else:
        print("Loading dataset from remote")

    dataset_list = load_dataset(args, args.dataset, args.from_remote_data)
    dataset_train = datasets.concatenate_datasets(
        [d["train"] for d in dataset_list]
    ).shuffle(seed=42)
    if args.test_dataset:
        dataset_list = load_dataset(args, args.test_dataset, args.from_remote_data)
    dataset_test = datasets.concatenate_datasets([d["test"] for d in dataset_list])
    dataset = datasets.DatasetDict({"train": dataset_train, "test": dataset_test})
    dataset = dataset.map(
        partial(
            tokenize, args, tokenizer, prompt_in_label=True, return_text=return_text
        ),
        num_proc=args.num_workers,
    )

    print("original dataset length: ", len(dataset["train"]))
    dataset = dataset.filter(
        lambda x: not x["exceed_max_length"], num_proc=args.num_workers
    )
    print("filtered dataset length: ", len(dataset["train"]))

    if len(dataset["train"]) == 0 and len(dataset["test"]) == 0:
        raise ValueError("No samples found within the maximum token length.")

    dataset = dataset.remove_columns(
        ["instruction", "input", "output", "exceed_max_length"]
    )

    if not return_text:
        dataset.save_to_disk(cache_dataset_path)

    return dataset


def get_trainer(args, model, tokenizer, dataset, formatted_time):
    """
    Create the trainer and training arguments
    """

    common_args = {
        "run_name": args.run_name,
        "output_dir": os.path.join(
            args.working_dir, "finetuned_models", f"{args.run_name}_{formatted_time}"
        ),
        "num_train_epochs": args.num_epochs,
        # "dataloader_num_workers": args.num_workers,
        "remove_unused_columns": False,  # maybe remove
        # -------------------------------------------
        "report_to": "wandb",
        "logging_dir": os.path.join(args.working_dir, "logs"),
        "logging_steps": args.log_interval,
        "save_steps": args.eval_steps,
        "eval_steps": args.eval_steps,
        "evaluation_strategy": args.evaluation_strategy,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        # -------------------------------------------
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.scheduler,
        "load_best_model_at_end": args.load_best_model,
        "fp16": args.fp16 & torch.cuda.is_available(),
        "bf16": args.bf16 & torch.cuda.is_available(),
        "optim": args.optim,
        "gradient_accumulation_steps": args.gradient_steps,
        "resume_from_checkpoint": args.resume_from_checkpoint
        # "label_names":[]
    }

    if args.distributed:
        distributed_args = {
            # "deepspeed": args.ds_config,
            "per_device_train_batch_size": args.batch_size,
            "per_device_eval_batch_size": args.batch_size,
            "ddp_find_unused_parameters": False,
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
                target_modules=lora_module_dict["mamba"],
                task_type=TaskType.CAUSAL_LM,
                lora_dropout=0.1,
                bias="none",
            )
        elif "pythia" in args.base_model:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                # lora_alpha=32,
                lora_alpha=args.lora_r,
                lora_dropout=0.1,
                target_modules=lora_module_dict["pythia"],
                bias="none",
            )
        trainer = custom_training.CustomSFTTrainer(
            model=model,
            # tokenizer=tokenizer,
            args=training_args,
            peft_config=peft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=custom_training.CustomDataCollatorSeq2Seq(
                tokenizer, padding=True
            ),
            prompt_loss_weight=args.prompt_loss_weight,
            max_seq_length=args.max_length,  # just to keep the warning silent as we are handling this ourselves
            tokenized_datasets=True,  # the secret sauce to make this work
        )

        # update args for logging
        common_args.update(**peft_config.__dict__)
    else:
        trainer = custom_training.CustomSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=custom_training.CustomDataCollatorSeq2Seq(
                tokenizer, padding=True
            ),
            prompt_loss_weight=args.prompt_loss_weight,
            max_seq_length=args.max_length,  # just to keep the warning silent as we are handling this ourselves
            tokenized_datasets=True
        )

        # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     peft_config=lora_config,
    #     train_dataset=dataset,
    #     dataset_text_field="quote",
    # )
    return trainer, training_args, common_args
