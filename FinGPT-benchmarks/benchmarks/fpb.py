import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from log_dtos import ClsMetrics
from typing import Tuple, List

from tqdm import tqdm
import datasets
import torch
from torch.utils.data import DataLoader
from functools import partial

dic = {
        0:"negative",
        1:'neutral',
        2:'positive',
    }

with open('sentiment_templates.txt') as f:
    templates = [l.strip() for l in f.readlines()]
    

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

       
def vote_output(x):
    output_dict = {'positive': 0, 'negative': 0, 'neutral': 0} 
    for i in range(len(templates)):
        pred = change_target(x[f'out_text_{i}'].lower())
        output_dict[pred] += 1
    if output_dict['positive'] > output_dict['negative']:
        return 'positive'
    elif output_dict['negative'] > output_dict['positive']:
        return 'negative'
    else:
        return 'neutral'
    
def test_fpb(args, model, tokenizer, prompt_fun=None, silent=True) -> Tuple[Dataset | DatasetDict, ClsMetrics]:
    # print what test is being done
    print("Testing on Financial Phrasebank dataset")

    batch_size = args.batch_size
    # instructions = load_dataset("financial_phrasebank", "sentences_50agree")
    instructions = load_from_disk("../data/financial_phrasebank-sentences_50agree/")
    instructions = instructions["train"]
    instructions = instructions.train_test_split(seed = 42)['test']
    instructions = instructions.to_pandas()
    instructions.columns = ["input", "output"]
    instructions["output"] = instructions["output"].apply(lambda x:dic[x])

    if prompt_fun is None:
        instructions["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis = 1)
    
    instructions[["context","target"]] = instructions.apply(format_example, axis = 1, result_type="expand")
    # print example
    if not silent:
        print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")

    context = instructions['context'].tolist()
    
    total_steps = instructions.shape[0]//batch_size + 1
    if not silent:
        print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, return_token_type_ids=False)
        for k in tokens.keys():
            tokens[k] = tokens[k].to(model.device)
        res = model.generate(**tokens, max_length=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        if not silent:
            print(f'{i}: {res_sentences[0]}')
        # So this is problematic, but better than the original alternative:
        # `out_text = [o.split("Answer: ")[1] if "Answer: " in o else "" for o in res_sentences]``
        # out_text_list += out_text
        # what if the answer has multiple answers? Then this restricts it to the
        # first answer only -> we are keeping original behaviour, just making sure
        # that when the model does not return an answer, we don't don't crash on the
        # indexig
        # TODO: see if this chanes the numbers?
        for answer in res_sentences:
            # if both
            if "Answer: " in answer:
                # and
                out_text = answer.split("Answer: ")
                if len(out_text) >= 2:
                    # then
                    out_text_list.append(out_text[1])
                    continue
                # otherwise in any case
                out_text_list.append("")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    instructions["out_text"] = out_text_list
    instructions["new_target"] = instructions["target"].apply(change_target)
    instructions["new_out"] = instructions["out_text"].apply(change_target)

    acc = accuracy_score(instructions["new_target"], instructions["new_out"])
    f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average = "macro")
    f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average = "micro")
    f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average = "weighted")

    print()
    print("*"*10)
    print("FPB")
    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")
    print("*"*10)
    print()

    return instructions, ClsMetrics(acc, f1_macro, f1_micro, f1_weighted)


def test_fpb_mlt(args, model, tokenizer, silent=True) -> Tuple[Dataset | DatasetDict, List[ClsMetrics]]:
    print("Running test_fpb_mlt")
    batch_size = args.batch_size
    # instructions = load_dataset("financial_phrasebank", "sentences_50agree")
    dataset = load_from_disk('../data/financial_phrasebank-sentences_50agree/')
    dataset = dataset["train"]#.select(range(300))
    dataset = dataset.train_test_split(seed=42)['test']
    dataset = dataset.to_pandas()
    dataset.columns = ["input", "output"]
    dataset["output"] = dataset["output"].apply(lambda x: dic[x])
    dataset["text_type"] = dataset.apply(lambda x: 'news', axis=1)
    
    dataset["output"] = dataset["output"].apply(change_target)
    dataset = dataset[dataset["output"] != 'neutral']
    
    out_texts_list = [[] for _ in range(len(templates))]
    
    def collate_fn(batch):
        inputs = tokenizer(
            [f["context"] for f in batch], return_tensors='pt',
            padding=True, max_length=args.max_length,
            return_token_type_ids=False
        )
        return inputs
    
    for i, template in enumerate(templates):
        dataset = dataset[['input', 'output', "text_type"]]
        dataset["instruction"] = dataset['text_type'].apply(lambda x: template.format(type=x) + "\nOptions: positive, negative")
        # dataset["instruction"] = dataset['text_type'].apply(lambda x: template.format(type=x) + "\nOptions: negative, positive")
        dataset[["context", "target"]] = dataset.apply(format_example, axis=1, result_type="expand")
        
        dataloader = DataLoader(Dataset.from_pandas(dataset), batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

        log_interval = len(dataloader) // 5

        for idx, inputs in enumerate(tqdm(dataloader)):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            res = model.generate(**inputs, do_sample=False, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id)
            res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
            if not silent:
                tqdm.write(f'{idx}: {res_sentences[0]}')
            # So this is problematic, but better than the original alternative:
            # `            out_text = [o.split("Answer: ")[1] for o in res_sentences]
            #              out_texts_list[i] += out_text`
            # out_text_list += out_text
            # what if the answer has multiple answers? Then this restricts it to the
            # first answer only -> we are keeping original behaviour, just making sure
            # that when the model does not return an answer, we don't don't crash on the
            # indexig
            # TODO: see if this chanes the numbers?
            for answer in res_sentences:
                # if both
                if "Answer: " in answer:
                    # and
                    out_text = answer.split("Answer: ")
                    if len(out_text) >= 2:
                        # then
                        out_texts_list[i].append(out_text[1])
                        continue
                    # otherwise in any case
                    out_texts_list[i].append("")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    for i in range(len(templates)):
        dataset[f"out_text_{i}"] = out_texts_list[i]
        dataset[f"out_text_{i}"] = dataset[f"out_text_{i}"].apply(change_target)
    
    dataset["new_out"] = dataset.apply(vote_output, axis=1, result_type="expand")
    dataset.to_csv('tmp.csv')

    print()
    print("*"*10)
    print("FPB")
    print("*"*10)
    
    metrics = []
    for k in [f"out_text_{i}" for i in range(len(templates))] + ["new_out"]:

        acc = accuracy_score(dataset["target"], dataset[k])
        f1_macro = f1_score(dataset["target"], dataset[k], average="macro")
        f1_micro = f1_score(dataset["target"], dataset[k], average="micro")
        f1_weighted = f1_score(dataset["target"], dataset[k], average="weighted")
        
        print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")
        print("*"*10)
        metrics.append(ClsMetrics(acc, f1_macro, f1_micro, f1_weighted))
    
    print()

    return dataset, metrics