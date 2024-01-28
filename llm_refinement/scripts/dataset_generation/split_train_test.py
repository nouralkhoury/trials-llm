from sklearn.model_selection import train_test_split
from utils.jsons import dump_json, load_json
import argparse


def main():
    parser = argparse.ArgumentParser(description="""Generate train/test sets
                                     from previously annotated dataset in JSON 
                                     format.""")

    parser.add_argument(
        "--annotated",
        required=True,
        help="Path to the annotated trials in JSON format")

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to save train and test JSON files")

    parser.add_argument(
        "--train-perc",
        required=False,
        default=70,
        type=int,
        help="Train set size percentage. Default is 70%")

    parser.add_argument(
        "--random-state",
        required=False,
        default=42,
        type=int,
        help="Random state for train_test_split(). Default is 42"
    )

    args = parser.parse_args()

    annotated = load_json(args.annotated)

    # convert dict to list of dict
    list_annotated = [{'trial_id': trial_id, **trial_data} for trial_id, trial_data in annotated.items()]

    # Get size of train from %
    train_size = int(len(list_annotated) * args.train_perc/100)
    training_data, test_data = train_test_split(list_annotated,
                                                train_size=train_size,
                                                random_state=args.random_state)

    dump_json(data={"size": len(training_data), "ids": training_data},
              file_path=f"{args.output_dir}/train_set.json")

    dump_json(data={"size": len(test_data), "ids": test_data},
              file_path=f"{args.output_dir}/test_set.json")


if __name__ == "__main__":
    main()
