"""
This script reads two JSON files containing structured data and
merges the records within the 'data' key. Then, it stratifies the merged
records based on class labels derived from the 'inclusion' and 'exclusion' keys
Finally, it creates a stratified sample of the records and saves the result to
an output JSON file.

Usage:
python merge_and_stratify.py --files <path_to_first_JSON_file> <path_to_second_JSON_file> --output-file <output_JSON_file>

Arguments:
    --files: Path to two annotated dataset JSON files.
    --output-file: Output directory to save the stratified JSON file.
"""
import random
import argparse
from utils.jsons import load_json, dump_json


def merge_data_structures(structure1, structure2):
    """
    Merges the records within the 'data' key of two data structures.

    Args:
        structure1 (dict): First data structure.
        structure2 (dict): Second data structure.

    Returns:
        dict: Merged data structure with a 'size' key and a merged 'data' list.
              Returns None if data structures are invalid for merging.
    """
    if 'data' in structure1 and 'data' in structure2 and isinstance(structure1['data'], list) and isinstance(structure2['data'], list):
        # Merge the records within the "data" key
        merged_data = structure1['data'] + structure2['data']
        merged_structure = {
            'size': len(merged_data),
            'data': merged_data
        }
        return merged_structure
    else:
        print("Invalid data structures for merging.")
        return None


def assign_biomarker_class(record):
    """
    Assign a biomarker class label to a record based on inclusion and
    exclusion.

    Args:
        record (dict): A record containing 'inclusion' and 'exclusion' lists.

    Returns:
        str: Biomarker class label ('No Biomarker', 'Both', 'Only Inclusion','Only Exclusion').
    """
    if not record['inclusion'] and not record['exclusion']:
        record_class = 'No Biomarker'
    elif record['inclusion'] and record['exclusion']:
        record_class = 'Both'
    elif record['inclusion']:
        record_class = 'Only Inclusion'
    elif record['exclusion']:
        record_class = 'Only Exclusion'
    return record_class


def create_stratified_sample(records, sample_size):
    """
    Create a stratified sample from records based on biomarker class labels.

    Args:
        records (dict): Data structure containing 'data' list with records.
        sample_size (int): Total size of the stratified sample.

    Returns:
        dict: Stratified sample data structure with a 'size' key and a sampled 'data' list.
    """
    class_records = {}
    for record in records['data']:
        record_class = assign_biomarker_class(record=record)
        class_records.setdefault(record_class, []).append(record)
    sample_size_per_class = sample_size // len(class_records)
    sampled_records = []
    for class_label, class_list in class_records.items():
        random.seed(42)
        sampled_class = random.sample(class_list, min(sample_size_per_class, len(class_list)))
        sampled_records.extend(sampled_class)
    result = {
        'size': len(sampled_records),
        'data': sampled_records
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="""Generate random biomarkers
                                     from Civic data and query them in a
                                     ChromaDB collection.""")
    parser.add_argument("--files",
                        nargs=2,
                        required=True,
                        help="Paths to the two JSON files")
    parser.add_argument("--output-file",
                        required=True,
                        help="Output directory to save stratified JSON file")

    args = parser.parse_args()
    json_files = [load_json(file_path) for file_path in args.files]
    merged_records = merge_data_structures(*json_files)
    stratified_sample_data = create_stratified_sample(merged_records, 50)
    dump_json(stratified_sample_data, args.output_file)


if __name__ == "__main__":
    main()
