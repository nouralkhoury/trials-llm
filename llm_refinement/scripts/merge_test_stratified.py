
import random
import argparse
from utils.jsons import load_json, dump_json


def merge_sets(structure1, structure2):
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


def assign_class_label(record):
    if not record['inclusion'] and not record['exclusion']:
        record_class = 'A'  # no biomarker
    elif record['inclusion'] and record['exclusion']:
        record_class = 'B'  # both inclusion and exclusion
    elif record['inclusion']:
        record_class = 'C'  # only inclusion
    elif record['exclusion']:
        record_class = 'D'  # only exclusion
    return record_class


def stratified_sample(records, sample_size):
    class_records = {}
    for record in records['data']:
        record_class = assign_class_label(record=record)
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
    parser = argparse.ArgumentParser(description="Generate random biomarkers from Civic data and query them in a ChromaDB collection.")
    parser.add_argument("--file-2", required=True, help="Path to the first JSON file")
    parser.add_argument("--file-1", required=True, help="Path to the second JSON file")
    parser.add_argument("--output-file", required=True, help="Output directory to save stratified JSON file")

    args = parser.parse_args()
    json_1 = load_json(args.file_1)
    json_2 = load_json(args.file_2)
    merged_records = merge_sets(json_1, json_2)
    sample = stratified_sample(merged_records, 50)
    dump_json(sample, args.output_file)


if __name__ == "__main__":
    main()
