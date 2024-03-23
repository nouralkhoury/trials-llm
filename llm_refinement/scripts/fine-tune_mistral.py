#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Mistral-7B to Extract Biomarkers from Clincical Trials

# In[2]:


# # You only need to run this once per machine
# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q -U datasets scipy ipywidgets matplotlib --user


# ## Load train and validation set

# In[3]:


# First we authorize huggingface
#!huggingface-cli login


# In[4]:


from datasets import load_dataset

dataset = load_dataset('nalkhou/clinical-trials', split=['train', 'validation', 'test'])

train_dataset = dataset[0]
validation_dataset = dataset[1]
test_dataset = dataset[2]

print("Train dataset:")
print(train_dataset)
print("\nValidation dataset:")
print(validation_dataset)
print("\Test dataset:")
print(test_dataset)


# ## Accelerator

# In[5]:


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


# ## Load Base Model

# In[6]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
base_model_id = "NousResearch/Hermes-2-Pro-Mistral-7B"
#base_model_id = "mlabonne/AlphaMonarch-7B"


# In[7]:


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")


# ## Tokenization

# ### Formatting prompt

# In[8]:


def formatting_func(example):
    text = f"""<|im_start|>system
You are a helpful assistant that extracts only genomic biomarkers from the supplied clinical trial data and responds in JSON format. Here's the json schema you must adhere to:<schema> {{"inclusion_biomarker": [[]], "exclusion_biomarker": [[]]}}</schema>
In this context, limit the extraction of genomic biomarkers to the following categories: gene alteration (mutation, fusion, rearrangement, copy number alteration, deletion, insertion, translocation), pathway alterations, gene expression, protein expression, pathway expression, HLA, TMB (tumor molecular burden, TMB-H or TMB-L), MSI (microsatellite instability, MSI-H, MSI-L, MSS, microsatellite stable) status, gene pathway alteration like dMMR (deficient Mismatch Repair Pathway) or pMMR (proficient Mismatch Repair), and protein status (HER2, ER, PgR, PD-L1).

Do not extract non-genomic biomarkers, which refer to any indicators not directly related to genetic or genomic information. Ignore information such as age, medical conditions, potential pregnancy, disease stage, allergies, treatment history, drugs, therapies, treatment, histology, and tumor cancer types, diseases, HIV, infections, and more. Also, ignore information about levels, scores, doses, expression ratios, and illnesses. Do not consider biomarkers related to model experimental animals, historical data, or previous studies.

Preserve logical connections (AND, OR) between genomic biomarkers. Group 'AND'-linked genomic biomarkers in the same list, and place 'OR'-linked genomic biomarkers in separate lists. Treat main bullets in "Inclusion Criteria" as AND logic, and "Exclusion Criteria" as OR logic, unless specified otherwise. Handle ambiguous logic in the sentence as OR.

Ensure each genomic biomarker is a string with the gene name preceding the variant. Remove the words "gene", "allele", "status", and "mutation" (when a specific variant is given). Make the variant singular and noun-based. Replace "mutant" with "mutation". Include a space between the gene name, its variant if they are connected. Include a space between the hormone name and its status if they are connected. Replace "positive expression" with "expression" and symbols "-" and "+" with "negative" and "positive" respectively, except in MSI status or known fusions separated by "-". Add "germline" or "somatic" terms in parentheses at the end of the corresponding biomarker. Ignore biomarkers mentioned as "exceptions" or after "other than". Handle synonyms in parentheses by extracting the genomic biomarker but ignoring the synonym. Extract each genomic biomarker once. Expand the genomic biomarkers when needed.

To summarize, extract only genomic biomarkers from the supplied clinical trial data, focusing on the categories mentioned above. Ignore any non-genomic biomarkers and unrelated information such as age, medical conditions, treatment history, cancer, drugs, therapies, histology, levels and scores. If no genomic biomarkers are found, return empty lists in JSON. Do not make assumptions or add biomarkers. Ensure output list contains only lists of strings when there exist genomic biomarkers in the input, following this example: {{"inclusion_biomarker": [["GeneA variantA"], ["GeneX variantY]], "exclusion_biomarker": []}}. Do not repeat a genomic biomarker.<|im_end|>
<|im_start|>user
Extract the genomic biomarker from the clinical trial below. Just generate the JSON object without explanation.
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
    return text


# Set up the tokenizer. Add padding on the left as it makes training use less memory.
# 
# 

# In[6]:


#!pip install sentencepiece


# In[7]:


tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    # add_prefix_space = False
)
tokenizer.pad_token = tokenizer.eos_token

def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))


# In[8]:


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = validation_dataset.map(generate_and_tokenize_prompt)


# ## Fine-tuning

# ### Set Up LoRA

# Now, to start our fine-tuning, we have to apply some preprocessing to the model to prepare it for training. For that use the prepare_model_for_kbit_training method from PEFT.

# In[18]:


from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# In[19]:


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# Here we define the LoRA config.
# 
# `r` is the rank of the low-rank matrix used in the adapters, which thus controls the number of parameters trained. A higher rank will allow for more expressivity, but there is a compute tradeoff.
# 
# `alpha` is the scaling factor for the learned weights. The weight matrix is scaled by `alpha/r`, and thus a higher value for alpha assigns more weight to the LoRA activations.
# 
# The values used in the QLoRA paper were `r=64` and `lora_alpha=16`, and these are said to generalize well, but we will use `r=32` and `lora_alpha=64` so that we have more emphasis on the new fine-tuned data while also reducing computational complexity.

# In[21]:


from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=2,
    lora_alpha=4,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.1,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)


# See how the model looks different now, with the LoRA adapters added:
# 
# 

# In[ ]:


### Weights and biases
#!pip install -q wandb -U

import wandb, os
os.environ["WANDB_PROJECT"] = "ctrials-finetune"
os.environ["WANDB_API_KEY"] = "ea9f2686146ccde695a0b52255ed33f58ccafde5"
wandb.login()


# ### Run training

# In[24]:


if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True


# In[25]:


model = accelerator.prepare_model(model)


# In[26]:


import transformers
from datetime import datetime

project = "ctrials-finetune"
base_model_name = "mistral-7B"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=3,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5, # Want a small lr for finetuning
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,              # When to start reporting loss
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=10,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=10,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


# In[ ]:




