import argparse
from datasets import load_dataset
import pandas as pd
import json

# Dictionary with dataset names and their Hugging Face identifiers
datasets_info = {
    "FinGPT_fiqaqa": "FinGPT/fingpt-fiqa_qa",
    "FinGPT_convqa": "FinGPT/fingpt-convfinqa",
    # Add more datasets as needed
}

def download_and_process_dataset(name, identifier, split='train'):
    print(f"Processing {name} with {split} split...")
    dataset = load_dataset(identifier, split=split)

    df = dataset.to_pandas()

    # Check if required columns exist in selected dataset/s
    required_columns = ['instruction', 'input', 'output']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Dataset {name} is missing required columns: {', '.join(missing_columns)}. Please ensure dataset columns match.")
        return  # Skip further processing for this dataset

    df.to_csv(f"{name}_{split}.csv", index=False)
    print(f"Processed and saved {name}")

def merge_datasets_and_export_to_jsonl(dataset_names, limit_mb=500, split='train'):
    all_data_frames = []
    for name in dataset_names:
        try:
            df = pd.read_csv(f"{name}_{split}.csv")
        except FileNotFoundError:
            print(f"Skipped merging {name} due to missing file.")
            continue
        all_data_frames.append(df)
    merged_df = pd.concat(all_data_frames, ignore_index=True)

    byte_limit = limit_mb * 1024 * 1024

    with open("merged_data.jsonl", 'w') as file:
        for _, row in merged_df.iterrows():
            json_line = row.to_json() + "\n"
            next_size = file.tell() + len(json_line.encode('utf-8'))  # Predict next size
            if next_size > byte_limit:
                print(f"Approaching the size limit ({byte_limit} bytes). Current file size: {file.tell()} bytes. Stopping to prevent exceeding the limit.")
                break
            file.write(json_line)
            file.flush()  # Ensure the buffer is flushed, and file size is updated
        else:
            print(f"Finished writing data without reaching the size limit. Final file size: {file.tell()} bytes.")

def main():
    parser = argparse.ArgumentParser(description='Download, process, and merge datasets from Hugging Face.')
    parser.add_argument('--dataset', type=str, help='Name of the dataset to process', default=None)
    parser.add_argument('--size_limit', type=int, default=500, help='Maximum size of the final JSONL file in MB')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Dataset split to process')

    args = parser.parse_args()
    processed_datasets = []

    if args.dataset:
        if args.dataset in datasets_info:
            download_and_process_dataset(args.dataset, datasets_info[args.dataset], args.split)
            processed_datasets.append(args.dataset)
        else:
            print(f"No dataset found with the name {args.dataset}.")
    else:
        for name, identifier in datasets_info.items():
            download_and_process_dataset(name, identifier, args.split)
            processed_datasets.append(name)

    # Always attempt to merge if at least one dataset has been processed
    if processed_datasets:
        merge_datasets_and_export_to_jsonl(processed_datasets, args.size_limit, args.split)

if __name__ == "__main__":
    main()