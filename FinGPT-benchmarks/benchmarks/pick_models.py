import os
import re

def get_name(input_string):
    # Define the regular expression pattern
    pattern = re.compile(r"((?:mamba|pythia)_(?:s|ms|m|l)_(?:mt\+|mt|mqsq)_\d(?:_b)?(?:_mt_\d)?)[\d_]+$")
    
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
    checkpoint_regex = re.compile(r'checkpoint-(\d{4})$')
    
    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path):
            final_best_path = os.path.join(full_path, 'final-best')
            adapter_config_path = os.path.join(full_path, 'adapter_config.json')
            safetensors_files = [f for f in os.listdir(full_path) if f.endswith('.safetensors')]
            
            # Check for 'final-best' directory
            if os.path.exists(final_best_path):
                directories_info[entry] = final_best_path
            # Check for 'adapter_config.json' or any '.safetensors' files
            elif os.path.exists(adapter_config_path) or safetensors_files:
                directories_info[entry] = entry
            else:
                # Search for checkpoint directories
                checkpoints = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d)) and checkpoint_regex.match(d)]
                if checkpoints:
                    # Find the checkpoint folder with the highest number
                    highest_checkpoint = max(checkpoints, key=lambda x: int(checkpoint_regex.match(x).group(1)))
                    highest_checkpoint_path = os.path.join(full_path, highest_checkpoint)
                    directories_info[entry] = highest_checkpoint_path
                    
    return directories_info

# Specify the base directory to search in
base_directory = "/home/ubuntu/repos/mlp/training/finetuned_models"  # Change this to your specific path
directories_info = ls_models(base_directory)
for dir_name, path in directories_info.items():
    print(f'"{dir_name}": "{path}"')

print(f"Found {len(directories_info)} models:")

for i, k in enumerate(directories_info.keys()):
    print(f"{i}\t{get_name(k)}")

