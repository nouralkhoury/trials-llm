"""
This script generates random biomarkers from the Civic dataset, queries them in a ChromaDB collection, 
and saves the results for training and testing purposes.

The script takes the paths to the ChromaDB collection persist directory (--persist-dir) 
and the Civic dataset file (--civic-path) as command-line arguments.

Usage:
python script_name.py --persist-dir /path/to/persist_dir --civic-path /path/to/civic_data.csv
"""

import pandas as pd
import chromadb
from chromadb.config import Settings
from sklearn.model_selection import train_test_split
import random
import json
import argparse


def load_collection(collection_name, persist_dir):
    """
    Load the collection of clinical trials from the ChromaDB database.

    Parameters:
        - collection_name (str): The name of the collection to load, e.g., 'ctrials'.
        - persist_dir (str): The path to the directory where the ChromaDB collection is persisted.

    Returns:
        - chromadb.Collection: The ChromaDB collection containing clinical trial data.
    """
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_dir
    ))
    collection = client.get_collection(collection_name)
    return collection

def get_civic_biomarkers(civic):
    """
    Extract unique biomarkers from the given CIViC (Clinical Interpretations of Variants in Cancer) dataset.

    Parameters:
        - civic (pd.DataFrame): The CIViC dataset containing gene and variant information.

    Returns:
        - numpy.ndarray: An array of unique biomarkers formed by concatenating gene and variant columns.
    """
    civic['biomarkers'] = civic['gene'] + " " + civic['variant']
    return civic['biomarkers'].unique()


def get_random_nums(seed, input_size, output_size):
    """
    Generate a list of random numbers sampled without replacement from a given range.

    Parameters:
    - seed (int): The seed value for the random number generator.
    - input_size (int): The size of the range from which to sample random numbers.
    - output_size (int): The number of random numbers to generate.

    Returns:
    - list[int]: A list of unique random numbers sampled without replacement.
    """
    random.seed(seed)
    return random.sample(range(0, input_size), output_size)


def generate_random_data(civic_path, persist_dir, size=250):
    """
    Generate random data by querying a ChromaDB collection with a randomly selected set of biomarkers.

    Parameters:
        - civic_path (str): The file path to the Civic dataset containing gene and variant information.
        - persist_dir (str): The path to the directory where the ChromaDB collection is persisted.
        - size (int, optional): The number of random biomarkers to select. Default is 250.

    Returns:
        - dict: A dictionary containing query results with randomly selected biomarkers.
        The dictionary has keys 'ids' and 'documents', where 'ids' is a list of document IDs,
        and 'documents' is a list of corresponding document content.
    """
    # load collection
    trials = load_collection('ctrials', persist_dir)
    # Get civic biomarkers list
    civic = pd.read_csv(civic_path)
    biomarkers = get_civic_biomarkers(civic) 
    # Generate the random biomarkers list
    random_numbers = get_random_nums(24, len(biomarkers), size)
    selected_biomarkers = biomarkers[random_numbers]
    results = trials.query(query_texts=selected_biomarkers, 
                           n_results=1,
                           include = [ "documents" ])
    return results


def save_data(results, file_name):
    with open(file_name,'w') as fi:
        json.dump(results, fi, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Generate random biomarkers from Civic data and query them in a ChromaDB collection.")
    parser.add_argument("--persist-dir", required=True, help="Path to the ChromaDB collection persist directory")
    parser.add_argument("--civic-path", required=True, help="Path to the Civic data file")
    args = parser.parse_args()

    results = generate_random_data(args.civic_path, args.persist_dir)
    final_results = [{'id': id_val[0], 'prompt': doc_val[0]} for id_val, doc_val in zip(results['ids'], results['documents'])]
    training_data , test_data  = train_test_split(final_results, train_size=0.8)
    save_data({"size": len(final_results), "data": final_results}, "random_trials.json")
    save_data({"size": len(training_data), "data": training_data}, "random_train.json")
    save_data({"size": len(test_data), "data": test_data}, "random_test.json")

if __name__ == "__main__":
    main()