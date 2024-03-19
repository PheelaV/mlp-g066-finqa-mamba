from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM
from transformers.utils import logging

# this is for the pesky eror about padding to the left, I think that is incorrect
# as the models have been trained with right padding
# this thread suggests to supress the warning
# https://stackoverflow.com/questions/74748116/huggingface-automodelforcasuallm-decoder-only-architecture-warning-even-after
import torch
import argparse
import os
from log_dtos import ClsMetrics
import datetime
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
from peft import PeftModel
import sys
import os
from pathlib import Path
import re
import wandb
from wandb.apis.public import Run
import lm_eval
from lm_eval.logging_utils import WandbLogger

from collections import namedtuple


fake_args = namedtuple(
    "args",
    ["batch_size", "max_length", "logging", "base_model"],
)
# args.batch_size
# args.max_length

logging.get_logger("transformers").setLevel(logging.ERROR)

# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

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


# Specify the base directory to search in
base_directory = "~/repos/mlp/training/finetuned_models"


def get_name(input_string):
    # Define the regular expression pattern
    pattern = re.compile(
        r"((?:mamba|pythia)_(?:s|ms|m|l)_(?:mt\+|mt|mqsq)_\d(?:_b)?(?:_mt_\d)?)[\d_]+$"
    )

    # Search for a match in the input string
    match = pattern.search(input_string)

    # If a match is found, return the captured group
    if match:
        return match.group(1)  # Return the first (and only) capturing group
    else:
        return None


def ls_models(base_path):
    directories_info = {}

    # Regular expression to match 'checkpoint-XXXX' folders
    checkpoint_regex = re.compile(r"checkpoint-(\d{4})$")

    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path):
            final_best_path = os.path.join(full_path, "final-best")
            adapter_config_path = os.path.join(full_path, "adapter_config.json")
            safetensors_files = [
                f for f in os.listdir(full_path) if f.endswith(".safetensors")
            ]

            # Check for 'final-best' directory
            if os.path.exists(final_best_path):
                directories_info[entry] = final_best_path
            # Check for 'adapter_config.json' or any '.safetensors' files
            elif os.path.exists(adapter_config_path) or safetensors_files:
                directories_info[entry] = full_path
            else:
                # Search for checkpoint directories
                checkpoints = [
                    d
                    for d in os.listdir(full_path)
                    if os.path.isdir(os.path.join(full_path, d))
                    and checkpoint_regex.match(d)
                ]
                if checkpoints:
                    # Find the checkpoint folder with the highest number
                    highest_checkpoint = max(
                        checkpoints,
                        key=lambda x: int(checkpoint_regex.match(x).group(1)),
                    )
                    highest_checkpoint_path = os.path.join(
                        full_path, highest_checkpoint
                    )
                    directories_info[entry] = highest_checkpoint_path

    return directories_info


def main(args):
    run_info = {k: {"path": v} for k, v in ls_models(args.base_directory).items()}
    import wandb

    api = wandb.Api()
    runs = api.runs("mlp-24-g066/mlp-g066")

    filter = args.pos_filter
    negative_filter = args.neg_filter

    positive_filter = f".*{filter}.*"
    filter_regex_pos = re.compile(positive_filter)
    filter_regex_neg = re.compile(negative_filter)
    # filter_regex.search

    filter_pos_total = sum(1 for r in runs if not filter_regex_pos.match(r.name))
    filter_neg_total = sum(1 for r in runs if not filter_regex_neg.match(r.name))
    find_counter = 0
    filtered_counter = 0
    announcement = ""
    for r in runs:
        r: Run = r
        p = Path(r.config["output_dir"])
        run_name = p.name
        if run_name not in run_info:
            print(f"[not found] {run_name}")
            continue
        matched_run = run_info[run_name]
        run_info[run_name]["max_len"] = r.summary["max_len"]
        run_info[run_name]["model_name"] = get_name(run_name)
        find_counter += 1
        pos_filter_result = bool(filter_regex_pos.search(run_name))
        neg_filter_result = bool(filter_regex_neg.search(run_name))
        if not pos_filter_result or neg_filter_result:
            print(
                f"[filtered out] {run_name=}, pos:{not pos_filter_result}, neg: {neg_filter_result}"
            )
            run_info.pop(run_name)
            continue
        filtered_counter += 1
        print(
            f"model_name: {matched_run['model_name']};run_name: {run_name};max_len: {matched_run['max_len']}; path_exists: {os.path.exists(matched_run['path'])}"
        )
        if not args.dry_run:
            formatted_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
            r.summary["benchmark_run"] = formatted_time
            r.update()
    announcement = f"total: {len(run_info)}; found {find_counter}; negative filter{filter_neg_total}; positive filter {filter_pos_total}; total included {filtered_counter}"
    print(announcement)

    print("*" * 50)
    for run_name, run in run_info.items():
        # if args.peft_model:
        #     run_name += f"_peft_{args.peft_model.replace('/', '_')}"
        # if args.logging:

        # else:
        #     os.environ['WANDB_DISABLED'] = 'true'

        print("#" * 30)
        print("#" * 30)
        print(f"Running for {run_name}...")
        print(f"Path: {run['path']}")
        print(f"Task (GeneralEval): {args.task}")
        print(f"Dataset (FinEval): {args.dataset}")
        print(f"Batch factor: {args.batch_factor}")
        print("#" * 30)
        print("#" * 30)

        if args.dry_run:
            continue

        if not args.dry_run and (args.lm_eval or args.fin_eval):
            wandb_run = wandb.init(
                project="mlp-g066-benchmarks2",
                name=run_name,
                group=run_name,
            )

        if args.lm_eval:
            import wandb

            config = {}
            config.update(vars(args))

            wandb.summary["model_name"] = run["model_name"]
            wandb.summary["max_len"] = run["max_len"]
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={run['path']},trust_remote_code=True",
                # tasks="arc_challenge,arc_easy,lambada,hellaswag,piqa,winogrande",
                tasks=args.task.split(","),
                log_samples=True,
            )

            wandb_logger = WandbLogger(
                project="lm-eval-harness-integration", job_type="eval"
            )  # or empty if wandb.init(...) already called before
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            wandb_logger.log_eval_samples(results["samples"])

            # if args.force_use_model:
            #     model_name = args.base_model
            # elif args.from_remote:
            #     model_name = parse_model_name(args.base_model, args.from_remote)
            # else:
            #     model_name = "../" + parse_model_name(args.base_model)

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        if "mamba" in run["model_name"]:
            model = MambaForCausalLM.from_pretrained(run["path"])
        else:
            model = AutoModelForCausalLM.from_pretrained(run["path"])
        model = model.to(device)
        model.eval()

        # if args.peft_model is not None:
        #     base_model = model
        #     model = PeftModel.from_pretrained(base_model, args.peft_model, device_map="auto")

        model = model.eval()
        # model.model_parallel = True
        # exit()
        func_args = fake_args(
            32 * args.batch_factor, run["max_len"], args.logging, run["model_name"]
        )

        tokenizer = get_tokenizer(func_args, run["path"])
        print(f"pad: {tokenizer.pad_token_id}, eos: {tokenizer.eos_token_id}")
        # args.batch_size
        # args.max_length
        if args.fin_eval:
            with torch.no_grad():
                for data in args.dataset.split(","):
                    if data == "fpb":
                        _, metrics = test_fpb(func_args, model, tokenizer)
                        log_cls_metrics(func_args, data, metrics)
                    elif data == "fpb_mlt":
                        _, template_metrics = test_fpb_mlt(func_args, model, tokenizer)
                        for i, metrics in enumerate(template_metrics):
                            log_cls_metrics(metrics, data, i)
                    elif data == "fiqa":
                        _, metrics = test_fiqa(func_args, model, tokenizer)
                        log_cls_metrics(func_args, data, metrics)
                    elif data == "fiqa_mlt":
                        _, template_metrics = test_fiqa_mlt(func_args, model, tokenizer)
                        for i, metrics in enumerate():
                            log_cls_metrics(func_args, data, metrics, i)
                    elif data == "tfns":
                        _, metrics = test_tfns(func_args, model, tokenizer)
                        log_cls_metrics(func_args, data, metrics)
                    elif data == "nwgi":
                        _, metrics = test_nwgi(func_args, model, tokenizer)
                        log_cls_metrics(func_args, data, metrics)
                    elif data == "convfinqa":
                        _, acc = test_convfinqa(func_args, model, tokenizer)
                        wandb.summary[f"{data}_acc"] = acc
                        wandb.log({f"{data}_acc": acc})
                    elif data == "fineval":
                        _, acc = test_fineval(func_args, model, tokenizer)
                        wandb.summary[f"{data}_acc"] = acc
                        wandb.log({f"{data}_acc": acc})
                    elif data == "re":
                        metrics = test_re(func_args, model, tokenizer)
                        log_what_is_this(func_args, data, metrics)
                    # These two need to be looked at if we are to use them...
                    elif data == "headline":
                        test_headline(func_args, model, tokenizer)
                    elif data == "ner":
                        test_ner(func_args, model, tokenizer)
                    else:
                        raise ValueError("undefined dataset.")

        if not args.dry_run and (args.lm_eval or args.fin_eval):
            wandb_run.finish()

    print("*" * 30)
    print(announcement)

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
    wandb.log({f"{data}_acc" + postfix: acc})
    wandb.log({f"{data}_f1_macro" + postfix: f1_macro})
    wandb.log({f"{data}_f1_micro" + postfix: f1_micro})
    wandb.log({f"{data}_f1_weighted" + postfix: f1_weighted})


def log_what_is_this(args, metrics, data):
    if not args.logging:
        return

    import wandb

    (precision, recall, f1_score, precision_re, recall_re, f1_score_re) = metrics
    wandb.log({f"{data}_precision": precision})
    wandb.log({f"{data}_recall": recall})
    wandb.log({f"{data}_f1_score": f1_score})
    wandb.log({f"{data}_precision_re": precision_re})
    wandb.log({f"{data}_recall_re": recall_re})
    wandb.log({f"{data}_f1_score_re": f1_score_re})
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
        "--base_directory",
        required=True,
        default=base_directory,
        type=str,
    )
    parser.add_argument(
        "--pos_filter",
        default="",
        type=str,
    )
    parser.add_argument(
        "--neg_filter",
        default="^$",
        type=str,
    )
    parser.add_argument(
        "--task",
        default="arc_challenge,arc_easy,lambada,hellaswag,piqa,winogrande",
        type=str,
    )
    parser.add_argument(
        "--lm_eval",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="execute lm eval tasks",
    )
    parser.add_argument(
        "--fin_eval",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="execute FinGPT eval tasks",
    )
    parser.add_argument(
        "--batch_factor",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--logging",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="Whether to log or not",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="List what will be processed and end",
    )
    parser.add_argument("--instruct_template", default="default")
    parser.add_argument(
        "--from_remote", action=argparse.BooleanOptionalAction, default=False
    )

    args = parser.parse_args()

    main(args)
