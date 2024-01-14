"""
Manual Test Set Generation Script

This script generates a dataset from a list of manually selected clinical
trial IDs stored in a JSON file. It extracts clinical trial documents, IDs,
and initializes empty inclusion and exclusion lists. The results are saved in
a structured JSON format.

Usage:
    python script_name.py --persist-dir <CHROMADB_PERSIST_DIR> --collection <COLLECTION_NAME>
                         --input-file <INPUT_JSON_FILE> --output-dir <OUTPUT_DIR>

Parameters:
    --persist-dir: Path to the ChromaDB persist directory.
    --collection: ChromaDB collection name.
    --input-file: Path to the JSON file containing the IDs for the manually selected trials.
    --output-dir: Output directory to save the generated JSON file.
"""
import argparse
import logging
from modules.chromadb_handler import ChromaDBHandler
from utils.jsons import load_json, dump_json


logging.basicConfig(level=logging.ERROR)


def get_trials_data(collection, ids):
    """
    Retrieve data for a list of clinical trial IDs from the ChromaDB
    collection.

    Args:
        collection (ChromaDB collection): The ChromaDB collection.
        ids (list): List of clinical trial IDs.

    Returns:
        dict: Dictionary containing the size of the dataset and the extracted
        data.
    """

    results = {
        "size": len(ids),
        "data": []}

    for id in ids:
        trial = collection.get(ids=[id])
        doc = trial['documents'][0]
        results['data'].append({
                        "id": id,
                        "prompt": doc,
                        "inclusion": [],
                        "exclusion": []})
    return results


def main():
    parser = argparse.ArgumentParser(description="""Generate manual clinical
                                     trials.""")
    parser.add_argument("--persist-dir",
                        required=True,
                        help="Path to the ChromaDB persist directory"
                        )

    parser.add_argument("--collection",
                        required=True,
                        help="ChromaDB collection name")

    parser.add_argument("--input-file",
                        required=True,
                        help="""Path to the JSON file containing the ids for
                        the manually selected trials."""
                        )

    parser.add_argument("--output-file",
                        required=True,
                        help="Output path to JSON file")

    args = parser.parse_args()
    try:
        collection = ChromaDBHandler(
            persist_dir=args.persist_dir,
            collection_name=args.collection).collection
    except SystemExit:
        print("Exiting due to an error.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    ids = load_json(args.input_file)['ids']

    results = get_trials_data(collection=collection, ids=ids)

    dump_json(results, f"{args.output_file}")


if __name__ == "__main__":
    main()
