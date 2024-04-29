"""
Description: This script generates train and test sets from a previously annotated dataset in JSON format. It splits the dataset into train and test subsets, appends the clinical trial text for each id, and saves the resulting sets to JSON files.

Usage:
python script_name.py --annotated [path_to_annotated_json] --output-dir [output_directory] --collection_name [collection_name] --persist_dir [persist_directory] [--train-perc train_percentage] [--random-state random_state]

Arguments:
- --annotated: Path to the annotated trials in JSON format.
- --output-dir: Output directory to save train and test JSON files.
- --train-perc: Train set size percentage. Default is 70%.
- --random-state: Random state for train_test_split(). Default is 42.
- --collection_name: Name of the ChromaDB collection.
- --persist_dir: Persist directory of the ChromaDB collection.

Note:
Ensure that the ChromaDB collection is properly configured and accessible, and that the annotated dataset follows the required format for processing.
"""

from sklearn.model_selection import train_test_split
from utils.jsons import dump_json, load_json
from modules.chromadb_handler import ChromaDBHandler

import argparse


def main():
    parser = argparse.ArgumentParser(description="""Generate train/test sets
                                     from previously annotated dataset in JSON
                                     format.""")

    parser.add_argument(
        "--annotated",
        required=True,
        help="Path to the annotated trials in JSON format")

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to save train and test JSON files")

    parser.add_argument(
        "--train-perc",
        required=False,
        default=70,
        type=int,
        help="Train set size percentage. Default is 70%")

    parser.add_argument(
        "--random-state",
        required=False,
        default=42,
        type=int,
        help="Random state for train_test_split(). Default is 42"
    )
    parser.add_argument("--collection_name", type=str, required=True, help="Name of the collection.")
    parser.add_argument("--persist_dir", type=str, required=True, help="Persist directory of the collection.")

    args = parser.parse_args()
    # Load JSON file
    try:
        annotated = load_json(args.annotated)
    except Exception as e:
        print(f"Error loading annotated JSON file: {e}")
        return

    collection = ChromaDBHandler(persist_dir=args.persist_dir, collection_name=args.collection_name).collection

    # convert dict to list of dict
    list_annotated = [{'trial_id': trial_id, "output": {"inclusion_biomarker": trial_data.get('inclusion_biomarker', []), "exclusion_biomarker": trial_data.get('exclusion_biomarker', [])}} for trial_id, trial_data in annotated.items()]
    train_size = int(len(list_annotated) * args.train_perc/100)

    # Iterate through the list_annotated and add the ChromaDB document for each trial
    for trial in list_annotated:
        trial_id = trial['trial_id']
        trial_document = collection.get(ids=[trial_id])['documents'][0]
        trial['document'] = trial_document

    # Split data into train and test
    try:
        training_data, test_data = train_test_split(list_annotated,
                                                    train_size=train_size,
                                                    random_state=args.random_state)
    except Exception as e:
        print(f"Error during train-test split: {e}")
        return

    # Save train and test sets to JSON files
    try:
        dump_json(data={"size": len(training_data), "ids": training_data},
                  file_path=f"{args.output_dir}/train_set.json")

        dump_json(data={"size": len(test_data), "ids": test_data},
                  file_path=f"{args.output_dir}/test_set.json")
    except Exception as e:
        print(f"Error saving train/test sets to JSON files: {e}")


if __name__ == "__main__":
    main()
