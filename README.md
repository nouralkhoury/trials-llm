# Trials LLM biomarker Predict

## Overview
This repository systematically investigates the impact of varying prompt engineering and fine-tuning strategies on different Large Language Models (LLMs) for the prediction of inclusion and exclusion genetic biomarkers. The study is conducted across diverse cancer clinical trials training dataset sizes to assess and optimize model performance.

## Get Started

### Environment Setup
Start by creating the environment and installing the necessary packages. This can be done by running `make install-llm`. Once the environment is set up, activate it by running `source venv-llm/bin/activate`.

### Create Dataset
To generate the train and test datasets, execute the script `scripts/data_generation/split_train_test.py`. Use the `data/interim/random_t_annotation_500_42.json` as the input file for this script.

### Make prediction

#### Zero-Shot and Few-Shot
To perform zero-shot or few-shot predictions on the test set, use the script `scripts/single_prompt_evaluation.py`. Ensure that you pass the correct arguments. The prompts required for this script can be found in `data/prompts`. You can use the following templates: `one-shot.json`, `zero-shot.json`, and `two-shot.json`.

#### Chain of Prompts
To run a chain of prompts, execute the script `scripts/chain_of_prompt_evaluation.py`. For a 2-chain pipeline, use `chain_1.json` and `chain_2.json`. For a 3-chain pipeline, use `chain_1.json`, `chain_interim.json`, and `chain_2.json`. The `chain_interim.json` serves as an intermediate chain between chain_1 and chain_2.
