README for Dataset Processing Script
====================================

Overview:
---------
This Python script downloads and merges datasets from Hugging Face, allowing for flexible dataset management for machine learning projects. It supports specifying dataset splits, enforcing a size limit on the final merged .jsonl file, and ensures datasets contain required columns.

Requirements:
-------------
- Python 3.6 or higher
- pandas library
- datasets library from Hugging Face

Installation of Dependencies:
-----------------------------
Ensure Python 3.6+ is installed on your system. You can install the required Python libraries using pip:

pip install pandas datasets


Usage:
------
The script can be run from the command line, with options to specify a single dataset, the dataset split ('train' or 'test'), and a size limit for the final .jsonl file.

To process a specific dataset with a custom size limit and split:

python script_name.py --dataset dataset1_name --split train --size_limit 300


Features:
---------
- **Dataset Selection**: Choose a specific dataset to process or process all predefined datasets.
- **Split Handling**: Select between 'train' and 'test' splits for dataset processing.
- **Size Limit**: Define a maximum size limit (in MB) for the final merged .jsonl file to prevent exceeding storage capacity.
- **Required Columns Check**: Ensures that each dataset contains 'instruction', 'input', and 'response' columns before processing.

Configuration:
--------------
Datasets to be processed are defined in the `datasets_info` dictionary within the script. Add or modify entries in this dictionary to include additional datasets from Hugging Face.

Troubleshooting:
----------------
- If the script indicates that the size limit has been reached prematurely, ensure that the specified size limit is appropriate for the volume of data being processed.
- Ensure datasets contain 'instruction', 'input', and 'response' columns. If these columns are missing or named differently, the script will notify the user and skip processing the dataset.

Support:
--------
For questions or issues, please check the Hugging Face documentation regarding dataset structure and the pandas library documentation for data manipulation queries.

