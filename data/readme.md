# Here be datasets

Mamba and Transformer (TBD) will require pre-training data for our Q&A in Finance. Here we should create a script that will accumulate our chosen datasets and concatonate them into our custom fine-tuning (training) dataset. There should also be a utility that will present the dataset in formats suitable for either of the models' use.


1. accumulate data
2. concatonate data
3. present data

- Beware of cross-pollination
- Pick relevant datasets for our task
- We will probably be using mostly hugging face machinery (Trainer from tranformers package, datasets package etc.)
- For Mamba an example of a suitable dataset/format is in the [mamba\_chat repository](https://github.com/havenhq/mamba-chat), if in doubts get in touch with Filip
- For the transformer model get in touch with Thomas
- yati yati yata...
