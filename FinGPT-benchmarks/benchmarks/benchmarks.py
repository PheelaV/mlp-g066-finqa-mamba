from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
from log_dtos import ClsMetrics

from fpb import test_fpb, test_fpb_mlt
from fiqa import test_fiqa, test_fiqa_mlt
from tfns import test_tfns
from nwgi import test_nwgi
from headline import test_headline
from ner import test_ner
from convfinqa import test_convfinqa
from fineval import test_fineval
from finred import test_re
from utils import parse_model_name
import sys

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

sys.path.append("../")


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


def main(args):
    
    # TODO needs a look at
    run_id = args.base_model
    if args.peft_model:
        run_id += f"_peft_{args.peft_model.replace('/', '_')}"
    if args.logging:
        import wandb

        config = {}
        config.update(vars(args))
        wandb.init(
            project="mlp-g066-benchmarks",
            name=args.run_name if args.run_name is not None else run_id,
            config=config,
            # dir=args.working_dir,
            group=args.run_name if args.run_name is not None else run_id,
        )
    else:
         os.environ['WANDB_DISABLED'] = 'true'
        
    print(f"Running for {run_id}...")

    if args.force_use_model:
        model_name = args.base_model
    elif args.from_remote:
        model_name = parse_model_name(args.base_model, args.from_remote)
    else:
        model_name = "../" + parse_model_name(args.base_model)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    is_mamba = "mamba" in model_name
    if is_mamba:
        model = MambaLMHeadModel.from_pretrained(model_name, device=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device=device)

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     trust_remote_code= True if not args.force_use_model else None,
    #     device_map={": device"},
    #     # load_in_8bit=True
    #     # fp16=True
    # )
    model.model_parallel = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"
    if args.base_model == "qwen":
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|extra_0|>")
    if not tokenizer.pad_token or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    print(f"pad: {tokenizer.pad_token_id}, eos: {tokenizer.eos_token_id}")

    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=8,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     target_modules=lora_module_dict[args.base_model],
    #     bias='none',
    # )
    # model = get_peft_model(model, peft_config)
    # model.load_state_dict(torch.load(args.peft_model + '/pytorch_model.bin'))

    # uncomment when using peft!
    # model = PeftModel.from_pretrained(model, args.peft_model)
    # model = get_peft_model(model, peft_config)
    model = model.eval()

    with torch.no_grad():
        for data in args.dataset.split(","):
            if data == "fpb":
                _, metrics = test_fpb(args, model, tokenizer)
                log_cls_metrics(args, data, metrics)
            elif data == "fpb_mlt":
                _, template_metrics = test_fpb_mlt(args, model, tokenizer)
                for i, metrics in enumerate(template_metrics):
                    log_cls_metrics(metrics, data, i)
            elif data == "fiqa":
                _, metrics = test_fiqa(args, model, tokenizer)
                log_cls_metrics(args, data, metrics)
            elif data == "fiqa_mlt":
                _, template_metrics = test_fiqa_mlt(args, model, tokenizer)
                for i, metrics in enumerate():
                    log_cls_metrics(args, data, metrics, i)
            elif data == "tfns":
                _, metrics = test_tfns(args, model, tokenizer)
                log_cls_metrics(args, data, metrics)
            elif data == "nwgi":
                _, metrics = test_nwgi(args, model, tokenizer)
                log_cls_metrics(args, data, metrics)
            elif data == "convfinqa":
                _, acc = test_convfinqa(args, model, tokenizer)
                wandb.summary[f"{data}_acc"] = acc
            elif data == "fineval":
                _, acc = test_fineval(args, model, tokenizer)
                wandb.summary[f"{data}_acc"] = acc
            elif data == "re":
                metrics = test_re(args, model, tokenizer)
                log_what_is_this(args, data, metrics)
            # These two need to be looked at if we are to use them...
            elif data == "headline":
                test_headline(args, model, tokenizer)
            elif data == "ner":
                test_ner(args, model, tokenizer)
            else:
                raise ValueError("undefined dataset.")

    print("Evaluation Ends.")


def log_cls_metrics(args, data, metrics: ClsMetrics, index=None):
    if not args.logging:
        return
    
    import wandb
    # I don't know how to log multiple metrics per dataset so I invendted the index
    # so that we may deterministically find what belongs to what later if we do need it...
    (acc, f1_macro, f1_micro, f1_weighted) = metrics
    postfix = f"_{index}" if index is not None else ""
    wandb.summary[f"{data}_acc" + postfix] = acc
    wandb.summary[f"{data}_f1_macro" + postfix] = f1_macro
    wandb.summary[f"{data}_f1_micro" + postfix] = f1_micro
    wandb.summary[f"{data}_f1_weighted" + postfix] = f1_weighted


def log_what_is_this(args, metrics, data):
    if not args.logging:
        return
    
    import wandb
    (precision, recall, f1_score, precision_re, recall_re, f1_score_re) = metrics
    wandb.summary[f"{data}_precision"] = precision
    wandb.summary[f"{data}_recall"] = recall
    wandb.summary[f"{data}_f1_score"] = f1_score
    wandb.summary[f"{data}_precision_re"] = precision_re
    wandb.summary[f"{data}_recall_re"] = recall_re
    wandb.summary[f"{data}_f1_score_re"] = f1_score_re


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument(
        "--base_model",
        required=True,
        type=str,
        # choices=[
        #     "chatglm2",
        #     "llama2",
        #     "llama2-13b",
        #     "llama2-13b-nr",
        #     "baichuan",
        #     "falcon",
        #     "internlm",
        #     "qwen",
        #     "mpt",
        #     "bloom",
        #     "pythia",
        #     ""
        # ],
        help=", ".join([
            "chatglm2",
            "llama2",
            "llama2-13b",
            "llama2-13b-nr",
            "baichuan",
            "falcon",
            "internlm",
            "qwen",
            "mpt",
            "bloom",
            "pythia",
        ]),
    )
    parser.add_argument("--peft_model", required=False, type=str)
    parser.add_argument("--run_name", default=None, type=str) 
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument(
        "--lora",
        action=argparse.BooleanOptionalAction,
        type=bool,
        help="Whether to use LoRA config for peft model (assumes a default configuration)",
    )
    parser.add_argument(
        "--logging",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="Whether to log or not)",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="The train batch size per device"
    )
    parser.add_argument("--instruct_template", default="default")
    parser.add_argument(
        "--from_remote", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--force_use_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force use of the model, this is a temporary measure...",
    )

    args = parser.parse_args()

    main(args)
