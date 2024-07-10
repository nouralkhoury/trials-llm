"""
Description: This script converts the test and train set from JSON to JSONL. Additionally, it splits the dataset into train and validation sets, and converts them into JSONL format. It provides an option to extend the training set with synthetic data.

Usage:
python script_name.py
"""

import hydra
import json
from utils.jsons import write_jsonl, to_jsonl
from sklearn.model_selection import train_test_split
from utils.jsons import load_json



@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    """
    Main function to split the dataset into train, validation, and test sets.
    """
    try:
        # Load datasets
        training_set = load_json(f"{cfg.data.interim_dir}/train_set.json")
        testing_set = load_json(f"{cfg.data.interim_dir}/test_set.json")

        # Extend training set with synthetic data if provided
        if cfg.synthetic_data:
            syn_data = load_json(f"{cfg.data.processed_dir}/gpt4_simulated_trials.json")
            training_set['ids'].extend(syn_data)  # Extend the training_set['ids'] list with syn_data

        # Convert datasets to JSONL format
        train_messages = to_jsonl(training_set['ids'])
        test_messages = to_jsonl(testing_set['ids'])

        # Split the training set into train and validation sets
        train_list, validation_list = train_test_split(train_messages, test_size=0.2, random_state=42)

        # Write data to JSONL files
        try:
            write_jsonl(f'{cfg.data.processed_dir}/ft_train.jsonl', train_list)
            write_jsonl(f'{cfg.data.processed_dir}/ft_validation.jsonl', validation_list)
            write_jsonl(f'{cfg.data.processed_dir}/ft_test.jsonl', test_messages)
        except json.JSONDecodeError as e:
            print(f"Error writing to file: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
