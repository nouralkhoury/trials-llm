import argparse
import json
from sklearn.model_selection import train_test_split
from utils.jsons import load_json


def write_jsonl(output_file, data_list):
    with open(output_file, 'w') as outfile:
        for entry in data_list:
            json.dump(entry, outfile)
            outfile.write('\n')


def to_jsonl(dataset):
    messages = []
    for trial in dataset:
        try:
            trial_id = trial.get('trial_id', None)
            if trial_id:
                if trial_id == "NCT04017130":  # skip this, outlier
                    continue
            document_key = 'document' if trial.get('document') else 'input'
            t_doc = trial[document_key]
            t_output = trial['output']

            current_message = {"input": t_doc, "output": t_output}

            messages.append(current_message)
        except (KeyError, IndexError) as e:
            print(f"Error processing trial: {trial['trial_id']} - {e}")
    return messages

def main(train_set, test_set, syn_set, output_dir):
    try:
        # load train
        training_set = load_json(train_set)
        testing_set = load_json(test_set)

        if syn_set:
            syn_data = load_json(syn_set)
            training_set['ids'].extend(syn_data)  # extend the training_set['ids'] list with syn_data

        train_messages = to_jsonl(training_set['ids'])
        test_messages = to_jsonl(testing_set['ids'])

        # Specify the test_size parameter to control the split ratio (e.g., 0.2 for 80-20 split)
        train_list, validation_list = train_test_split(train_messages, test_size=0.2, random_state=42)

        try:
            write_jsonl(f'{output_dir}/ft_withsyn_train.jsonl', train_list)
            write_jsonl(f'{output_dir}/ft_withsyn_validation.jsonl', validation_list)
            write_jsonl(f'{output_dir}/ft_test.jsonl', test_messages)
        except json.JSONDecodeError as e:
            print(f"Error writing to file: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split the dataset into train and validation sets.")
    parser.add_argument("--train-set", type=str, required=True, help="Path to train json.")
    parser.add_argument("--syn-set", type=str, required=False, default=None, help="Path to synthetically generated json.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the train and validation files.")
    parser.add_argument("--test-set", type=str, required=True, help="Output directory for the train and validation files.")

    args = parser.parse_args()
    main(args.train_set, args.test_set, args.syn_set, args.output_dir)
