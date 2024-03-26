import os
import sys
import time
import argparse
import progressbar
from pathlib import Path
from langchain.prompts import load_prompt
from modules.gpt_handler import GPTHandler
from modules.logging_handler import CustomLogger
from modules.chromadb_handler import ChromaDBHandler
from configurations.config import RESULTS_DATA
from utils.jsons import (
    load_json,
    flatten_lists_in_dict,
    dump_json,
    loads_json)
from utils.evaluation import evaluate_predictions, save_eval, get_metrics
from utils.token_count import num_tokens_from_string


def log_name(template_file, model):
    file_name_no_extension = Path(template_file).stem
    log_name = f"{model}_{file_name_no_extension}"
    return log_name


def load_prompt_file(file_path):
    """
    Check if the prompt file exists and load its content.

    Args:
        file_path (str): The path to the prompt file.

    Returns:
        str: The content of the prompt file if it exists, otherwise None.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    return load_prompt(file_path)


def input_args():
    parser = argparse.ArgumentParser(description="""Test LLM performance""")
    parser.add_argument("--test",
                        required=True,
                        help="Paths to JSON test set with annotation")

    parser.add_argument("--output-file",
                        required=True,
                        help="Output filename to save evaluation results")

    parser.add_argument("--prompt-1",
                        required=True,
                        help="Path to prompt JSON file with the template")

    parser.add_argument("--prompt-2",
                        required=True,
                        help="Path to prompt JSON file with the template")

    parser.add_argument("--prompt-interim",
                        required=False,
                        default=None,
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
    model, pl_tags, prompt_1, prompt_2, prompt_interim = args.model, args.tags, args.prompt_1, args.prompt_2, args.prompt_interim
    token_count = 0

    logger = CustomLogger(log_name("chain_of_prompts", model))

    try:
        test_set = load_json(args.test)
        print(f"Size of test set {args.test}: {test_set['size']}")
    except FileNotFoundError as e:
        logger.log_error(f"Failed to load {args.test}: {e}")
        sys.exit(1)

    try:
        # Load the first prompt file
        first_prompt_template = load_prompt_file(prompt_1)

        # Load the second prompt file
        second_prompt_template = load_prompt_file(prompt_2)

        # Load the third prompt file if provided
        interim_prompt_template = None
        if prompt_interim:
            interim_prompt_template = load_prompt_file(prompt_interim)

    except FileNotFoundError as e:
        logger.log_error(f"Template File {e.filename} does not exist: {e}")
        sys.exit(1)

    try:
        gpthandler = GPTHandler()
        # Set up LLM chain for the first prompt
        chain_1 = gpthandler.setup_gpt(
            model_name=model,
            prompt_template=first_prompt_template,
            pl_tags=pl_tags)

        # Set up LLM chain for the second prompt
        chain_2 = gpthandler.setup_gpt(
            model_name=model,
            prompt_template=second_prompt_template,
            pl_tags=pl_tags)

        # Set up LLM chain for the third prompt if provided
        chain_interim = None
        if interim_prompt_template:
            chain_interim = gpthandler.setup_gpt(
                model_name=model,
                prompt_template=interim_prompt_template,
                pl_tags=pl_tags)
    except Exception as e:
        logger.log_error(f"Failed to set up LLM chain: {e}")
        sys.exit(1)

    logger.log_info(f"Chain 1: {first_prompt_template.template}")
    if interim_prompt_template:
        logger.log_info(f"Chain interim: {interim_prompt_template.template}")
    logger.log_info(f"Chain 2: {second_prompt_template.template}")

    start_time = time.time()

    tp_inc, tn_inc, fp_inc, fn_inc = [], [], [], []
    tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf = [], [], [], []

    tp_ex, tn_ex, fp_ex, fn_ex = [], [], [], []
    tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf = [], [], [], []
    predicted_list, actual_list = [], []

    bar = progressbar.ProgressBar(maxval=test_set['size'], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    counter = 0
    for i in test_set['ids']:
        counter += 1
        bar.update(counter)
        try:
            trial_id = i['trial_id']
            logger.log_info(f"@ trial {trial_id}")

            actual = i['output']
            input_trial = i['document']

            response_1 = chain_1.run({'trial': input_trial})
            token_count += num_tokens_from_string(response_1, "cl100k_base") + num_tokens_from_string(input_trial, "cl100k_base") + num_tokens_from_string(first_prompt_template.template, "cl100k_base")

            if chain_interim:
                response_2 = chain_interim.run({'input_list': response_1})
                token_count += num_tokens_from_string(interim_prompt_template.template, "cl100k_base")
            else:
                response_2 = response_1

            response = chain_2.run({'input_list': response_2})
            token_count += num_tokens_from_string(response_2, "cl100k_base") + num_tokens_from_string(response, "cl100k_base") + num_tokens_from_string(second_prompt_template.template, "cl100k_base")

            try:
                response_parsed = loads_json(response)
            except Exception as e:
                logger.log_error(f"Trial {trial_id} Failed to parse JSON output: {e}")
                continue

            predicted_list.append(response_parsed)
            actual_list.append(actual)

            logger.log_info(f"Predicted: {response_parsed}")
            logger.log_info(f"Actual: {actual}")

            # Metrics
            evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl = compute_evals(response_parsed, actual)
            save_eval(tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf, evals_dnf_inclusion)
            save_eval(tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf, evals_dnf_exclusion)
            save_eval(tp_inc, tn_inc, fp_inc, fn_inc, evals_extract_incl)
            save_eval(tp_ex, tn_ex, fp_ex, fn_ex, evals_extract_exl)

            logger.log_info("\n")
        except Exception as e:
            logger.log_error(f"Trial {trial_id} Failed: {e}")

    end_time = time.time()
    latency = end_time - start_time
    logger.log_info(f"Latency: {latency} seconds")
    logger.log_info("\n\n\n")

    # Get Precision, recall, f1 score and accuracy
    inc = get_metrics(tp=sum(tp_inc), tn=sum(tn_inc), fp=sum(fp_inc), fn=sum(fn_inc))
    ex = get_metrics(tp=sum(tp_ex), tn=sum(tn_ex), fp=sum(fp_ex), fn=sum(fn_ex))
    inc_dnf = get_metrics(tp=sum(tp_inc_dnf), tn=sum(tn_inc_dnf), fp=sum(fp_inc_dnf), fn=sum(fn_inc_dnf))
    ex_dnf = get_metrics(tp=sum(tp_ex_dnf), tn=sum(tn_ex_dnf), fp=sum(fp_ex_dnf), fn=sum(fn_ex_dnf))
    results = {
        "Model": model,
        "Precited": predicted_list,
        "Actual": actual_list,
        "tp_inclusion": tp_inc,
        "fp_inclusion": fp_inc,
        "tn_inclusion": tn_inc,
        "fn_inclusion": fn_inc,
        "tp_exclusion": tp_ex,
        "tn_exclusion": tn_ex,
        "fp_exclusion": fp_ex,
        "fn_exclusion": fn_ex,
        "Inclusion Precision": [inc[0]],
        "Inclusion Recall": [inc[1]],
        "Inclusion F1": [inc[2]],
        "Inclusion Acc": [inc[3]],
        "Inclusion F2": [inc[4]],
        "Exclusion Precision": [ex[0]],
        "Exclusion Recall": [ex[1]],
        "Exclusion F1": [ex[2]],
        "Exclusion Acc": [ex[3]],
        "Exclusion F2": [ex[4]],
        "Inclusion DNF Precision": [inc_dnf[0]],
        "Inclusion DNF Recall": [inc_dnf[1]],
        "Inclusion DNF F1": [inc_dnf[2]],
        "Inclusion DNF Acc": [inc_dnf[3]],
        "Inclusion DNF F2": [inc_dnf[4]],
        "Exclusion DNF Precision": [ex_dnf[0]],
        "Exclusion DNF Recall": [ex_dnf[1]],
        "Exclusion DNF F1": [ex_dnf[2]],
        "Exclusion DNF Acc": [ex_dnf[3]],
        "Exclusion DNF F2": [ex_dnf[4]],
        "Latency (seconds)": latency,
        "tokens": token_count
    }
    bar.finish()
    output_file = os.path.join(RESULTS_DATA, args.output_file)
    try:
        # Read existing data from the file, if it exists
        existing_data = load_json(output_file)
    except FileNotFoundError:
        logger.log_error(f"Output file {output_file} does not exist")
        existing_data = {}

    # Append the new data to the existing results list
    if "results" in existing_data:
        existing_data["results"].append(results)
    else:
        existing_data["results"] = [results]

    dump_json(existing_data, output_file)


if __name__ == "__main__":
    main()
