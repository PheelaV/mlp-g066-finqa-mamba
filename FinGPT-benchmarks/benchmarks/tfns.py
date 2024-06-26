import warnings
warnings.filterwarnings("ignore")

from log_dtos import ClsMetrics
from typing import Tuple

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
import datasets
import torch

dic = {
    0:"negative",
    1:'positive',
    2:'neutral',
}

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

def test_tfns(args, model, tokenizer, prompt_fun=None, silent=True) -> Tuple[Dataset | DatasetDict, ClsMetrics]:
    # print what test is being done
    print("Testing on Twitter Financial News Sentiment dataset")

    batch_size = args.batch_size
    # dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
    dataset = load_from_disk('../data/twitter-financial-news-sentiment')
    dataset = dataset['validation']
    dataset = dataset.to_pandas()
    dataset['label'] = dataset['label'].apply(lambda x:dic[x])
    
    if prompt_fun is None:
        dataset["instruction"] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis = 1)

    dataset.columns = ['input', 'output', 'instruction']
    dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")

    # print example
    if not silent:
        print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()
    
    total_steps = dataset.shape[0]//batch_size + 1
    if not silent:
        print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512, return_token_type_ids=False)
        # tokens.pop('token_type_ids')
        for k in tokens.keys():
            tokens[k] = tokens[k].to(model.device)
        res = model.generate(**tokens, max_length=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
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

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")

    print()
    print("*"*10)
    print("FINEVAL")
    print("*"*10)
    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")
    print("*"*10)
    print()

    return dataset, ClsMetrics(acc, f1_macro, f1_micro, f1_weighted)