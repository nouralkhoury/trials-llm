import os
import torch
import argparse
import matplotlib.pyplot as plt
from utils.jsons import load_json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def configure_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def plot_data_lengths(tokenized_train_set, tokenized_test_set, dpi):
    plt.rc('xtick', labelsize=5) 
    plt.rc('ytick', labelsize=5)
    lengths = [len(x['input_ids']) for x in tokenized_train_set]
    lengths += [len(x['input_ids']) for x in tokenized_test_set]
    fig = plt.figure(figsize=(3.54,2.36), dpi=dpi) # specify figure size and resolution
    n, bins, patches = plt.hist(lengths, bins=90, linewidth=0.5, ec="black", color='C0')
    patches[len(patches)-1].set_fc('r') # color the last bar, it is the outlier
    plt.xlabel('Token count', fontsize=7) # Modify x-axis
    plt.ylabel('Frequency', fontsize=7) # Modify y-axis
    return plt


def main(train_set, test_set, outdir, model_id, fig_dpi):
    training_set = load_json(train_set)
    testing_set = load_json(test_set)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = configure_tokenizer(model_id)
    
    tokenized_train_set = [tokenizer(trial['document'] ) for trial in training_set['ids']]
    tokenized_test_set = [tokenizer(trial['document'] ) for trial in testing_set['ids']]

    plt = plot_data_lengths(tokenized_train_set, tokenized_test_set, fig_dpi)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, f'token_count_hist_redbar_{fig_dpi}.png'), bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the token count distribution')
    parser.add_argument('--trainset', type=str, help='Path to the JSON file containing the training samples')
    parser.add_argument('--testset', type=str, help='Path to the JSON file containing the test samples')
    parser.add_argument('--outdir', type=str, help='Output directory to save the plot')
    parser.add_argument('--model', type=str, help='Name of LLM model to be used for tokenization', default="NousResearch/Hermes-2-Pro-Mistral-7B")
    parser.add_argument('--dpi', type=int, help="dpi for image", default=400)
    args = parser.parse_args()
    main(args.trainset, args.testset, args.outdir, args.model, args.dpi)