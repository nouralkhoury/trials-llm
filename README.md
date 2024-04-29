# Trials LLM: LLM models applied for biomarker Extraction from cancer clinical trials

## Overview
Cancer remains a leading cause of death globally, underscoring the importance of clinical trials in advancing treatments. Precision medicine, reliant on genomic biomarkers, offers personalized therapies. Extracting patient eligibility information from vast unstructured text is challenging. Leveraging large language models (LLMs), this project focuses on extracting genetic biomarkers from cancer clinical trials. We explore the efficacy of LLMs, including gpt-3.5-turbo, gpt-4, and Hermes-2-Pro-Mistral-7B, employing various prompting techniques. Results demonstrate effective biomarker extraction, highlighting the importance of prompt selection over model size. The findings inform the selection of Hermes-2-Pro-Mistral-7B_DPO-155 as the most suitable model for this task.

This repository includes the source code, prompts, intermediate datasets and results for this project.

NOTE: The database that holds the collection of cancer clinical trials are not provided in this repo since they were generated in a previous project.

## Directory Structure for `llm_refinement`

- `data/`: Contains the interim dataset with manual annotations. Full datasets for testing and fine-tuning are available at [Hugging Face Model Hub](https://huggingface.co/nalkhou). This directory also includes the prompts used in the project.

- `scripts/dataset_generation/`: Python scripts for generating datasets and preparing them for DPO fine-tuning.

- `scripts/`: Contains scripts for fine-tuning and testing the models.

- `notebooks/`: Includes a notebook for performing exploratory data analysis (EDA) of the dataset.

- `modules/` and `utils/`: Houses classes and reusable functions necessary for the project's processes.

- `tests/`: Holds test units to evaluate whether the evaluation functions produce expected results.


## Get Started

### Environment Setup
Start by creating the environment and installing the necessary packages. This can be done by running `make install-llm`. Once the environment is set up, activate it by running `source venv-llm/bin/activate`.
These steps are only possible when the clinical trials collection is hosted locally.

### Creating Datasets

1. **random_generation.py**: This script queries clinical trials from our database based on a list of genomic biomarkers. It generates the file `random_trials_ids_{size}_{seed}.json`, containing a list of trial IDs with a possible inclusion of genomic biomarkers. The inclusion is not guaranteed but represents the closest match based on similarity between the query biomarker and the trial. Semantic search identifies relevant documents based on contextual similarity, prioritizing trials with the highest similarity to the query.

2. Manually annotate the `random_trials_ids_{size}_{seed}.json` file. The annotated file is provided in `data/interim` as `random_t_annotation_500_42.json`.

3. **split_train_test.py**: This script splits a previously annotated JSON dataset into train and test subsets. It appends the clinical trial text for each ID and saves the resulting sets as JSON files.

4. **simulated_data_gpt4.py**: This script generates simulated clinical trial data based on provided examples using the OpenAI API. It then extracts genomic biomarkers from the generated text and saves the simulated data as a JSON file.

5. **create_jsonL.py**: This script converts JSON datasets to JSONL format, splits them into train and validation sets, and extends the training set with synthetic data if required.

6. **generate_negatives.py**: This script generates negative outputs (rejected completions) for a given prompt using a pre-trained Language Model (LM). The LM generates responses, which are considered as negative examples, and stores them in a JSONL file. This negative completion is necessary for DPO fine-tuning. Alternatively, one can skip the previous steps and run this script directly on [Hugging Face Dataset Hub](https://huggingface.co/datasets/nalkhou/clinical-trials/) or [Hugging Face Dataset Hub](https://huggingface.co/datasets/nalkhou/clinical-trial-mixed) to generate the dataset required for finetuning.


### Fine-tuning

`dpo_file_train.py`: This script is used to fine-tune Hermes-2-Pro-Mistral-7B using DPO and QLoRA (Quantized Low-Rank Adapters) with the appropriate dataset previously generated.

### Evaluation

To evaluate the GPT models with zero-shot prompting or few-shot prompting, one can run the `scripts/single_prompt_evaluation.py` script with the corresponding prompt files:
- `data/prompts/zero-shot.json`
- `data/prompts/one-shot.json`
- `data/prompts/two-shot.json`

To evaluate the GPT models with a chain of prompts, one can run `scripts/chain_of_prompt_evaluation.py` using the prompt files:
- `data/prompts/chain_1.json`
- `data/prompts/chain_2.json`

To evaluate the open-source models, including the base model and the fine-tuned models provided at [Hugging Face Model Hub](https://huggingface.co/nalkhou/Hermes-2-Pro-Mistral-7B_DPO-92) and [Hugging Face Model Hub](https://huggingface.co/nalkhou/Hermes-2-Pro-Mistral-7B_DPO-156), one can run `scripts/test_hermes.py`.

