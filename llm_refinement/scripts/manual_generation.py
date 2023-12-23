import json
import argparse
from modules.chromadb_handler import ChromaDBHandler


def get_trials_data(collection):
    ids = [
            "NCT05061550", "NCT05062278", "NCT05065398", "NCT05053802",
            "NCT04293796", "NCT04222335", "NCT04222972", "NCT04241731",
            "NCT04670679", "NCT04461808", "NCT04536077", "NCT03817268",
            "NCT03475121", "NCT03390686", "NCT05354388", "NCT05430802",
            "NCT05241873", "NCT05639751", "NCT04771520"
            ]

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
    parser = argparse.ArgumentParser(description="Generate manual clinical trials.")
    parser.add_argument("--persist-dir",
                        required=True,
                        help="Path to the ChromaDB persist directory"
                        )

    parser.add_argument("--collection",
                        required=True,
                        help="ChromaDB collection name")

    parser.add_argument("--output-dir",
                        required=True,
                        help="Output directory to JSON file")

    args = parser.parse_args()

    collection = ChromaDBHandler(
        persist_dir=args.persist_dir,
        collection_name=args.collection).collection

    results = get_trials_data(collection=collection)

    with open(f"{args.output_dir}/manual_test_set.json", "w") as f:
        json.dump(results, f, indent=4)
