import os
import shutil
from datasets import load_dataset

import pathlib
# delete data if it exists

# just so we can execute this from any directory
local_file_folder = pathlib.Path(__file__).parent.absolute()

file_paths = [os.path.join(local_file_folder, path) for path in ("twitter-financial-news-sentiment", "news_with_gpt_instructions", "financial_phrasebank-sentences_50agree", "fiqa-2018")]


for file_path in file_paths:
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
        
# Load and save the datasets
dataset = load_dataset("pauri32/fiqa-2018")
dataset.save_to_disk(os.path.join(local_file_folder, "fiqa-2018"))

dataset = load_dataset("financial_phrasebank", "sentences_50agree")
dataset.save_to_disk(os.path.join(local_file_folder, "financial_phrasebank-sentences_50agree"))

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
dataset.save_to_disk(os.path.join(local_file_folder, "twitter-financial-news-sentiment"))

dataset = load_dataset("oliverwang15/news_with_gpt_instructions")
dataset.save_to_disk(os.path.join(local_file_folder, "news_with_gpt_instructions"))