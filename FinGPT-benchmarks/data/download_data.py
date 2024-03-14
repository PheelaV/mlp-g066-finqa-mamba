import os
import shutil
from datasets import load_dataset

# delete data if it exists
file_paths = ['twitter-financial-news-sentiment', 'news_with_gpt_instructions', 'financial_phrasebank-sentences_50agree', 'fiqa-2018']

for file_path in file_paths:
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
        
# Load and save the datasets
dataset = load_dataset('pauri32/fiqa-2018')
dataset.save_to_disk('fiqa-2018')

dataset = load_dataset('financial_phrasebank', 'sentences_50agree')
dataset.save_to_disk('financial_phrasebank-sentences_50agree')

dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
dataset.save_to_disk('twitter-financial-news-sentiment')

dataset = load_dataset('oliverwang15/news_with_gpt_instructions')
dataset.save_to_disk('news_with_gpt_instructions')