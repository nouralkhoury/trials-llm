import os
import sys
import time
import argparse
import progressbar
from pathlib import Path
from statistics import mean
from langchain.prompts import load_prompt
from modules.gpt_handler import GPTHandler
from modules.chromadb_handler import ChromaDBHandler
from modules.logging_handler import CustomLogger
from configurations.config import (
    CTRIALS_COLLECTION,
    PERSIST_DIRECTORY,
    RESULTS_DATA
    )
from utils.jsons import (
    load_json,
    flatten_lists_in_dict,
    dump_json,
    loads_json)
from utils.evaluation import evaluate_predictions, save_eval


def actual_output(trial):
    actual_inclusion = trial['inclusion_biomarker']
    actual_exclusion = trial['exclusion_biomarker']
    actual = {'inclusion_biomarker': actual_inclusion,
              'exclusion_biomarker': actual_exclusion}
    return actual


def log_name(template_file, model):
    file_name_no_extension = Path(template_file).stem
    log_name = f"{model}_{file_name_no_extension}"
    return log_name


def input_args():
    parser = argparse.ArgumentParser(description="""Test LLM performance""")
    parser.add_argument("--test",
                        required=True,
                        help="Paths to JSON test set with annotation")
    parser.add_argument("--output-file",
                        required=True,
                        help="Output filename to save evaluation results")
    parser.add_argument("--prompt",
                        required=True,
                        help="Path to prompt JSON file with the template")
    parser.add_argument("--model",
                        required=False,
                        help="OPENAI model name to be used",
                        default="gpt-3.5-turbo")
    parser.add_argument("--tags",
                        required=False,
                        help="List of PromptLayer tags",
                        nargs='+',
                        default=[])
    args = parser.parse_args()
    return args


def compute_evals(response, actual):
    # Evaluate results with DNF
    evals_dnf_inclusion = evaluate_predictions(response, actual, 'inclusion_biomarker')
    evals_dnf_exclusion = evaluate_predictions(response, actual, 'exclusion_biomarker')

    # Evaluate results without DNF
    flat_response = flatten_lists_in_dict(response)
    flat_actual = flatten_lists_in_dict(actual)
    evals_extract_incl = evaluate_predictions(flat_response, flat_actual, "inclusion_biomarker")
    evals_extract_exl = evaluate_predictions(flat_response, flat_actual, "exclusion_biomarker")

    return evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl


def main():
    args = input_args()
    model, template_file, pl_tags = args.model, args.prompt, args.tags

    logger = CustomLogger(log_name(template_file, model))

    try:
        test_set = load_json(args.test)
        print(f"Size of test set {args.test}: {test_set['size']}")
    except FileNotFoundError as e:
        print(f"Failed to load {args.test}: {e}")
        sys.exit(1)

    try:
        # check if template file exists
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"The file '{template_file}' does not exist.")
        prompt_template = load_prompt(template_file)
    except FileNotFoundError as e:
        print(f"Template File {template_file} does not exist: {e}")
        sys.exit(1)

    # set up LLM chain
    try:
        gpthandler = GPTHandler()
        llm_chain = gpthandler.setup_gpt(model_name=model,
                                         prompt_template=prompt_template,
                                         pl_tags=pl_tags)
    except Exception as e:
        print(f"Failed to set up GPTHandler {e}")
        sys.exit(1)

    # loading collection
    try:
        trials = ChromaDBHandler(PERSIST_DIRECTORY, CTRIALS_COLLECTION).collection
        print(f"Number of Trials in {CTRIALS_COLLECTION} collection: {trials.count()}")
    except Exception as e:
        print(f"Failed to load ChromaDB collection {CTRIALS_COLLECTION} from {PERSIST_DIRECTORY}: {e}")

    logger.log_info(f"Prompt Template: {prompt_template.template}")
    start_time = time.time()

    prec_inc, recall_inc, accs_inc, f1s_inc = [], [], [], []
    prec_inc_dnf, recall_inc_dnf, accs_inc_dnf, f1s_inc_dnf = [], [], [], []

    prec_ex, recall_ex, f1s_ex, accs_ex = [], [], [], []
    prec_ex_dnf, recall_ex_dnf, f1s_ex_dnf, accs_ex_dnf = [], [], [], []

    predicted_list, actual_list = [], []

    bar = progressbar.ProgressBar(maxval=test_set['size'], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    counter = 0
    for i in test_set['ids'][0:1]:
        counter += 1
        bar.update(counter+1)
        try:
            trial_id = i['trial_id']
            logger.log_info(f"@ trial {trial_id}")

            input_trial = trials.get(ids=[trial_id])['documents'][0]

            response = llm_chain({'trial': input_trial})
            try:
                response['text']
            except Exception as e:
                print(f"Trial {trial_id} Failed to generate text output: {e}")
                continue

            response_parsed = loads_json(response['text'])
            actual = actual_output(i)
            predicted_list.append(response_parsed)
            actual_list.append(actual)

            logger.log_info(f"Predicted: {response_parsed}")
            logger.log_info(f"Actual: {actual}")

            # Metrics
            evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl = compute_evals(response_parsed, actual)
            save_eval(prec_inc_dnf, recall_inc_dnf, f1s_inc_dnf, accs_inc_dnf, evals_dnf_inclusion)
            save_eval(prec_ex_dnf, recall_ex_dnf, f1s_ex_dnf, accs_ex_dnf, evals_dnf_exclusion)
            save_eval(prec_inc, recall_inc, f1s_inc, accs_inc, evals_extract_incl)
            save_eval(prec_ex, recall_ex, f1s_ex, accs_ex, evals_extract_exl)

            inc_status = True if evals_dnf_inclusion[3] == 1 else False
            exc_status = True if evals_dnf_exclusion[3] == 1 else False
            logger.log_info(f"Inclusion: {inc_status}")
            logger.log_info(f"Exclusion: {exc_status}")
            logger.log_info("\n")
        except Exception as e:
            print(f"Trial {trial_id} Failed: {e}")

    end_time = time.time()
    latency = end_time - start_time
    logger.log_info(f"Latency: {latency} seconds")
    logger.log_info("\n\n\n")
    results = {
        "prompt": prompt_template.template,
        "Model": model,
        "Precited": predicted_list,
        "Actual": actual_list,
        "precision_inclusion": prec_inc,
        "precision_inclusion_dnf": prec_inc_dnf,
        "precision_exclusion": prec_ex,
        "precision_exclusion_dnf": prec_ex_dnf,
        "recall_inclusion": recall_inc,
        "recall_inclusion_dnf": recall_inc_dnf,
        "recall_exclusion": recall_ex,
        "recall_exclusion_dnf": recall_ex_dnf,
        "Average Inclusion Precision": mean(prec_inc),
        "Average Inclusion Recall": mean(recall_inc),
        "Average Inclusion F1": mean(f1s_inc),
        "Average Inclusion Acc": mean(accs_inc),
        "Average Exclusion Precision": mean(prec_ex),
        "Average Exclusion Recall": mean(recall_ex),
        "Average Exclusion F1": mean(f1s_ex),
        "Average Exclusion Acc": mean(accs_ex),
        "Average Inclusion DNF Precision": mean(prec_inc_dnf),
        "Average Inclusion DNF Recall": mean(recall_inc_dnf),
        "Average Inclusion DNF F1": mean(f1s_inc_dnf),
        "Average Inclusion DNF Acc": mean(accs_inc_dnf),
        "Average Exclusion DNF Precision": mean(prec_ex_dnf),
        "Average Exclusion DNF Recall": mean(recall_ex_dnf),
        "Average Exclusion DNF F1": mean(f1s_ex_dnf),
        "Average Exclusion DNF Acc": mean(accs_ex_dnf),
        "Latency (seconds)": latency,
    }
    bar.finish()
    output_file = os.path.join(RESULTS_DATA, args.output_file)
    try:
        # Read existing data from the file, if it exists
        existing_data = load_json(output_file)
    except FileNotFoundError:
        print(f"Output file {output_file} does not exist")
        existing_data = {}

    # Append the new data to the existing results list
    if "results" in existing_data:
        existing_data["results"].append(results)
    else:
        existing_data["results"] = [results]

    dump_json(existing_data, output_file)


if __name__ == "__main__":
    main()
