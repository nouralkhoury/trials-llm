"""
Description: This script generates train and test sets from a previously annotated dataset in JSON format. It splits the dataset into train and test subsets, appends the clinical trial text for each id, and saves the resulting sets to JSON files.

Usage:
python script_name.py --annotated [path_to_annotated_json] --output-dir [output_directory] [--train-perc train_percentage] [--random-state random_state]

Arguments:
- --annotated: Path to the annotated trials in JSON format.
- --output-dir: Output directory to save train and test JSON files.
- --train-perc: Train set size percentage. Default is 70%.
- --random-state: Random state for train_test_split(). Default is 42.
"""

from sklearn.model_selection import train_test_split
from utils.jsons import dump_json, load_json
import hydra


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg):
    # Load JSON file
    try:
        annotated_path = f"{cfg.data.interim_dir}/random_t_annotation_500_42.json"
        annotated = load_json(annotated_path)
    except Exception as e:
        print(f"Error loading annotated JSON file: {e}")
        return


    # convert dict to list of dict
    list_annotated = [{'trial_id': trial_id, "output": {"inclusion_biomarker": trial_data.get('inclusion_biomarker', []), "exclusion_biomarker": trial_data.get('exclusion_biomarker', [])}, "document": trial_data['document']} for trial_id, trial_data in annotated.items()]
    train_size = int(len(list_annotated) * cfg.split_params.train_percent/100)

    # Split data into train and test
    try:
        training_data, test_data = train_test_split(list_annotated,
                                                    train_size=train_size,
                                                    random_state=cfg.split_params.random_state)
    except Exception as e:
        print(f"Error during train-test split: {e}")
        return

    # Save train and test sets to JSON files
    try:
        dump_json(data={"size": len(training_data), "ids": training_data},
                  file_path=f"{cfg.data.interim_dir}/train_set.json")

        dump_json(data={"size": len(test_data), "ids": test_data},
                  file_path=f"{cfg.data.interim_dir}/test_set.json")
    except Exception as e:
        print(f"Error saving train/test sets to JSON files: {e}")


if __name__ == "__main__":
    main()
