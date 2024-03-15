from seqeval.metrics import classification_report
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
from log_dtos import ClsMetrics
import datasets
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from functools import partial
import re
import sys
import numpy as np
sys.path.append('../')
from utils import *
    

relations = [
    'product_or_material_produced',
    'manufacturer',
    'distributed_by',
    'industry',
    'position_held',
    'original_broadcaster',
    'owned_by',
    'founded_by',
    'distribution_format',
    'headquarters_location',
    'stock_exchange',
    'currency',
    'parent_organization',
    'chief_executive_officer',
    'director_/_manager',
    'owner_of',
    'operator',
    'member_of',
    'employer',
    'chairperson',
    'platform',
    'subsidiary',
    'legal_form',
    'publisher',
    'developer',
    'brand',
    'business_division',
    'location_of_formation',
    'creator',
]


def cvt_text_to_pred(ref, text):
    
    preds = []
    for pred_txt in text.strip('.').split(';'):
        pred_match = re.match(r'^(.*):(.*),(.*)$', pred_txt)
        if pred_match is not None:
            relation, word1, word2 = pred_match.group(1).strip(), pred_match.group(2).strip(), pred_match.group(3).strip()
            if relation in relations and word1 in ref and word2 in ref:
                preds.append((relation, word1, word2))
            else:
                print("Not found Error: ", relation, word1, word2, ref)    
        else:
            print("Parse Error: ", pred_txt)
            
    return preds


def map_output(feature):

    ref = feature['input']
    label = cvt_text_to_pred(ref, feature['output'])
    pred = cvt_text_to_pred(ref, feature['out_text'])
    
    return {'label': label, 'pred': pred}


def calc_metric(gt_list, pred_list):
    # Initialize variables for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for (ground_truth, predicted_relations) in zip(gt_list, pred_list):
        # Calculate true positives, false positives, and false negatives
        for relation in predicted_relations:
            if relation in ground_truth:
                true_positives += 1
            else:
                false_positives += 1

        for relation in ground_truth:
            if relation not in predicted_relations:
                false_negatives += 1

    # Calculate precision, recall, and F1-Score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
    

def test_re(args, model, tokenizer, silent=True) -> Tuple[Dataset | DatasetDict, ClsMetrics]:

    dataset = load_from_disk('../data/fingpt-finred-re')['test']#.select(range(50))
    dataset = dataset.train_test_split(0.2, seed=42)['test']
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
        # res = model.generate(**inputs, max_length=args.max_length, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128)
        res = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id)
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
    dataset = dataset.to_pandas()
    
    if not silent:
        print(dataset)
    dataset.to_csv('tmp.csv')
    
    label = [[tuple(t) for t in d.tolist()] for d in dataset['label']]
    pred = [[tuple(t) for t in d.tolist()] for d in dataset['pred']]
    precision, recall, f1_score =  calc_metric(label, pred)
    
    label_re = [[t[0] for t in d.tolist()] for d in dataset['label']]
    pred_re = [[t[0] for t in d.tolist()] for d in dataset['pred']]
    precision_re, recall_re, f1_score_re =  calc_metric(label_re, pred_re)
    print()
    print("*"*10)
    print("FINRED")
    print(f"Precisions: {precision}, Recalls: {recall}, F1-Scores: {f1_score}")

    print("*"*10)
    print(f"RE? Precisions: {precision_re}, Recalls: {recall_re}, F1-Scores: {f1_score_re}")
    print("*"*10)
    print()
    
    return dataset, (precision, recall, f1_score, precision_re, recall_re, f1_score_re)