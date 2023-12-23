import json
import argparse
from modules.chromadb_handler import ChromaDBHandler


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

    parser.add_argument("--output-dir",
                        required=True,
                        help="Output directory to JSON file")

    args = parser.parse_args()

    collection = ChromaDBHandler(
        persist_dir=args.persist_dir,
        collection_name=args.collection).collection

    with open(args.input_file, "r") as f:
        ids = json.load(f)['ids']

    results = get_trials_data(collection=collection, ids=ids)

    with open(f"{args.output_dir}/manual_test_set.json", "w") as f:
        json.dump(results, f, indent=4)
