"""
This script generates random biomarkers from the Civic dataset,
queries them in a ChromaDB collection,
and saves the trial IDS returned for training and testing purposes.

Parameters:
    --persist-dir: Path to the ChromaDB persist directory
    --civic-path: Path the CIViC variant Summaries TSV file
    --output-dir: Path to directory to save the JSON files

Usage:
python script_name.py --persist-dir persist_dir --civic-path civic_data.csv --output-dir output_dir
"""
import random
import argparse
import pandas as pd
from utils.jsons import dump_json
from modules.chromadb_handler import ChromaDBHandler


def get_civic_biomarkers(civic):
    """
    Extract unique biomarkers from the given CIViC (Clinical Interpretations
    of Variants in Cancer) dataset.

    Parameters:
        - civic (pd.DataFrame): The CIViC dataset containing gene
        and variant information.

    Returns:
        - numpy.ndarray: An array of unique biomarkers formed by concatenating
        gene and variant columns.
    """
    civic['biomarkers'] = civic['gene'] + " " + civic['variant']
    return civic['biomarkers'].unique()


def get_random_nums(seed, input_size, output_size):
    """
    Generate a list of random numbers sampled without replacement
    from a given range.

    Parameters:
    - seed (int): The seed value for the random number generator.
    - input_size (int): The size of the range from which to sample random numbers.
    - output_size (int): The number of random numbers to generate.

    Returns:
    - list[int]: A list of unique random numbers sampled without replacement.
    """
    random.seed(seed)
    return random.sample(range(0, input_size), output_size)


def generate_random_data(civic_path, persist_dir, size=500, seed=42):
    """
    Generate random data by querying a ChromaDB collection with a randomly
    selected set of biomarkers.

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
    trials = ChromaDBHandler(persist_dir, 'ctrials').collection
    # Get civic biomarkers list
    civic = pd.read_csv(civic_path)
    biomarkers = get_civic_biomarkers(civic)
    # Generate the random biomarkers list
    random_numbers = get_random_nums(seed, len(biomarkers), size)
    selected_biomarkers = biomarkers[random_numbers]
    results = trials.query(query_texts=selected_biomarkers,
                           n_results=1,
                           include=[])  # return example {'ids': [['NCT04489433']], 'embeddings': None, 'documents': None, 'metadatas': None, 'distances': None}
    return results


def main():
    parser = argparse.ArgumentParser(description="""Generate random biomarkers
                                     from Civic data and query them in a
                                     ChromaDB collection.""")
    parser.add_argument(
            "--persist-dir",
            required=True,
            help="Path to the ChromaDB collection persist directory")

    parser.add_argument(
        "--civic-path",
        required=True,
        help="Path to the Civic data file")

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to save train and test JSON files")

    args = parser.parse_args()

    results = generate_random_data(args.civic_path, args.persist_dir)
    final_results = list(set([id_val[0] for id_val in results['ids']]))  # example ['NCT05252403', 'NCT05435248', 'NCT04374877']

    dump_json(data={"size": len(final_results), "ids": final_results},
              file_path=f"{args.output_dir}/random_trials_ids.json")


if __name__ == "__main__":
    main()
