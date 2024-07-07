"""
create_train_collection.py

Script to create a ChromaDB train collection from a subset of a full collection.

Usage:
python create_train_collection.py --train-collection <train_collection_name> --train-set-path <train_set_path> [--persist-train <persist_train_directory>]

Arguments:
--train-collection     Name of the train ChromaDB collection (default: "train-ctrials").
--train-set-path       Path to the train set JSON file (default: "data/processed/train_set.json").
--persist-train        ChromaDB Persist directory for the train collection (default: "data/collections/").

Example:
python create_train_collection.py --train-collection my_train_collection --train-set-path data/train_data.json --persist-train data/train_collections/

"""
import argparse
from modules.chromadb_handler import ChromaDBHandler
from utils.jsons import load_json
import os
import logging as log
from conf.config import CTRIALS_COLLECTION, PERSIST_DIRECTORY, PROCESSED_DATA


def create_train_collection(train_collection_name, train_set_path, persist_train):
    full_collection = ChromaDBHandler(PERSIST_DIRECTORY,
                                      CTRIALS_COLLECTION).collection

    print("Full collection loaded", full_collection.count())
    # Create or load the train collection
    db = ChromaDBHandler(persist_train,
                         train_collection_name)
    db_client = db.client
    # Load the old collection
    train_collection = db.collection
    print("Train collection loaded", train_collection.count())

    # Get train set
    train_set = load_json(train_set_path)

    print("Train set data loaded: ", len(train_set))

    print("Adding to collection")
    # Iterate through train set ids
    for train_id_info in train_set['ids']:
        trial_id = train_id_info['trial_id']
        if trial_id == "NCT04017130":  # skip outlier, large tokens
            continue
        # Get document from the full collection
        doc = full_collection.get(ids=[trial_id])['documents'][0]
        # Extract relevant information from the train set
        train_info = [i for i in train_set['ids'] if i['trial_id'] == trial_id][0]
        output_dict = {
            "inclusion_biomarker": train_info['inclusion_biomarker'],
            "exclusion_biomarker": train_info["exclusion_biomarker"]
        }
        # Convert output_dict to string
        output_dict_str = str(output_dict)
        # Add the document to the train collection
        train_collection.add(ids=[trial_id], documents=[doc], metadatas=[{'id': trial_id, 'output': output_dict_str}])
    print("Adding to collection... DONE!")
    db_client.persist()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a ChromaDB train collection from a subset of a full collection.")
    parser.add_argument("--persist-train",
                        required=False,
                        default="data/collection_train",
                        help="ChromaDB Persist directory")

    parser.add_argument("--train-collection",
                        required=False,
                        default="train-ctrials",
                        help="Name of the train ChromaDB collection")

    parser.add_argument("--train-set-path",
                        required=False,
                        default=os.path.join(PROCESSED_DATA, "train_set.json"),
                        help="Path to the train set JSON file")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    create_train_collection(args.train_collection,
                            args.train_set_path,
                            args.persist_train)
    log.info("Collection Created!")
