import os
import sys
import time
import argparse
import progressbar
from pathlib import Path
from langchain.prompts import load_prompt
from modules.gpt_handler import GPTHandler
from modules.logging_handler import CustomLogger
from conf.config import (
    RESULTS_DATA
)
from utils.jsons import (
    load_json,
    flatten_lists_in_dict,
    dump_json,
    loads_json)
from utils.evaluation import evaluate_predictions, save_eval, get_metrics


def log_name(template_file, model):
    file_name_no_extension = Path(template_file).stem
    log_name = f"{model}_{file_name_no_extension}"
    return log_name


def find_trial(k, trials):
    trial = next((t for t in trials if t['trial_id'] == k), None)
    if trial:
        return trial['document'], trial['output']
    else:
        return None, None


def input_args():
    parser = argparse.ArgumentParser(description="""Test LLM performance""")
    parser.add_argument("--test",
                        required=True,
                        help="Paths to JSON test set with annotation")
    # create a mutually exclusive group for the --n-example argument
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n-example", type=int, help="Number of examples for the Few shot.")
    group.add_argument("--no-n-example", action="store_true", help="Don't use few shot learning.")
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
    # add the --train argument to a separate argument group that is required only when --n-example is used
    train_group = parser.add_argument_group(title="required arguments when using --n-example")
    train_group.add_argument("--train", required=False, help="Paths to JSON train set with annotation")

    args = parser.parse_args()
    # check if --n-example was used and if so, make the --train argument required
    if args.n_example:
        if not args.train:
            parser.error("--train is required when using --n-example")
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
    model, template_file, pl_tags, n_examples = args.model, args.prompt, args.tags, args.n_example

    logger = CustomLogger(log_name(template_file, model))

    try:
        test_set = load_json(args.test)
        print(f"Size of test set {args.test}: {test_set['size']}")
    except FileNotFoundError as e:
        logger.log_error(f"Failed to load {args.test}: {e}")
        sys.exit(1)

    try:
        train_set = load_json(args.train)
        print(f"Size of test set {args.train}: {train_set['size']}")
    except FileNotFoundError as e:
        logger.log_error(f"Failed to load {args.train}: {e}")
        sys.exit(1)

    try:
        # check if template file exists
        if not os.path.exists(template_file):
            raise FileNotFoundError(f"The file '{template_file}' does not exist.")
        prompt_template = load_prompt(template_file)
    except FileNotFoundError as e:
        logger.log_error(f"Template File {template_file} does not exist: {e}")
        sys.exit(1)

    # set up LLM chain
    try:
        gpthandler = GPTHandler()
        llm_chain = gpthandler.setup_gpt(model_name=model,
                                         prompt_template=prompt_template,
                                         pl_tags=pl_tags)
    except Exception as e:
        logger.log_error(f"Failed to set up GPTHandler {e}")
        sys.exit(1)

    logger.log_info(f"Prompt Template: {prompt_template.template}")
    start_time = time.time()

    tp_inc, tn_inc, fp_inc, fn_inc = [], [], [], []
    tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf = [], [], [], []

    tp_ex, tn_ex, fp_ex, fn_ex = [], [], [], []
    tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf = [], [], [], []
    predicted_list, actual_list, failed_prediction = [], [], []

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

            if n_examples == 0:
                response = llm_chain({'trial': input_trial})
            else:
                example_id = "NCT03383575"
                example_doc, example_output = find_trial(example_id, train_set['ids'])
                example = f"""{example_doc}\nexample JSON:{example_output}"""
                if n_examples == 2:
                    example_id = "NCT05484622"
                    example_doc, example_output = find_trial(example_id, train_set['ids'])
                    example_2 = f"""{example_doc}\nJSON:{example_output}"""

                    response = llm_chain({'trial': input_trial, 'example': example, 'example2': example_2})
                else:
                    response = llm_chain({'trial': input_trial, 'example': example})
            try:
                response['text']
            except Exception as e:
                logger.log_error(f"Trial {trial_id} Failed to generate text output: {e}")
            try:
                response_parsed = loads_json(response['text'])
            except TypeError as e:
                logger.log_error(f"Trial {trial_id} Failed to parse text output: {e}")
                failed_prediction.append(response)
                if actual == {'inclusion_biomarker': [], 'exclusion_biomarker': []}:
                    evals_dnf_inclusion = evals_dnf_exclusion = evals_extract_incl = evals_extract_exl = (0,0,1,0)
                else:
                    response = {'inclusion_biomarker': [], 'exclusion_biomarker': []}
                    evals_dnf_inclusion, evals_dnf_exclusion, evals_extract_incl, evals_extract_exl = compute_evals(response, actual)
                save_eval(tp_inc_dnf, tn_inc_dnf, fp_inc_dnf, fn_inc_dnf, evals_dnf_inclusion)
                save_eval(tp_ex_dnf, tn_ex_dnf, fp_ex_dnf, fn_ex_dnf, evals_dnf_exclusion)
                save_eval(tp_inc, tn_inc, fp_inc, fn_inc, evals_extract_incl)
                save_eval(tp_ex, tn_ex, fp_ex, fn_ex, evals_extract_exl)
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
        "prompt": prompt_template.template,
        "Model": model,
        "correct_size": len(predicted_list),
        "failed_size": len(failed_prediction),
        "Precited": predicted_list,
        "Actual": actual_list,
        "Failed": failed_prediction,
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
