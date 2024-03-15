from seqeval.metrics import accuracy_score
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from typing import Tuple
from tqdm import tqdm
import datasets
import torch
from torch.utils.data import DataLoader
from functools import partial
import re
import sys
import numpy as np
sys.path.append('../')
from utils import *
    

def cvt_text_to_pred(text):
    if not text:
        return 'nan'
    pred_match = re.search(r'\d+(.\d+)', text)
    if pred_match is not None:
        pred = pred_match.group()
    else:
        print(text)
        pred = '0.0'
    return pred


def map_output(feature):

    label = cvt_text_to_pred(feature['output'])
    pred = cvt_text_to_pred(feature['out_text'])
    
    return {'label': label, 'pred': pred}


def test_convfinqa(args, model, tokenizer, silent=True) -> Tuple[Dataset | DatasetDict, float]:

    dataset = load_from_disk('../data/fingpt-convfinqa')['test']#.select(range(30))
    dataset = dataset.map(partial(test_mapping, args), load_from_cache_file=False)
    
    def collate_fn(batch):
        inputs = tokenizer(
            [f["prompt"] for f in batch], return_tensors='pt',
            padding=True, max_length=args.max_length,
            return_token_type_ids=False
        )
        return inputs
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    out_text_list = []
    log_interval = len(dataloader) // 5

    for idx, inputs in enumerate(tqdm(dataloader)):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        res = model.generate(**inputs, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        res_sentences = [tokenizer.decode(i, skip_special_tokens=True) for i in res]
        if (idx + 1) % log_interval == 0 and not silent:
            tqdm.write(f'{idx}: {res_sentences[0]}')
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
    
    dataset = dataset.add_column("out_text", out_text_list)
    dataset = dataset.map(map_output, load_from_cache_file=False)
    dataset = dataset.filter(lambda x: x['pred'] != 'nan')
    dataset = dataset.to_pandas()
    
    if not silent:
        print(dataset)
    dataset.to_csv('tmp.csv')
    
    label = [float(d) for d in dataset['label']]
    pred = [float(d) for d in dataset['pred']]
    acc = accuracy_score(label, pred)
    print()
    print("*"*10)
    print("convfinqa")
    print("*"*10)
    print('Accuracy: ', acc)
    print("*"*10)
    print()
    return dataset, acc