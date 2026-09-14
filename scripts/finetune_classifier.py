# Final LLM fine-tuning script for edema prediction using pre-processed radiology report texts
# Compiled by Ethan Phillips (Univ of Oxford) on 5 August 2024


#################################################################
# LOAD REQUIRED PACKAGES
import os
import sys
import numpy as np
import pandas as pd
import transformers
import os
import sys
import datasets
from datasets import Dataset
from pydantic import confloat, validate_call
from sklearn.model_selection import train_test_split
import torch
import evaluate
import wandb
import fire

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# set add one level higher folder to path (for accessing methods in other files)

# import processing method from data processing file
from helmet_mass_effect_pred.data.processing import (
    target_transform_partial,
    preprocess_rad_text,
    parse_dates,
    process_data,
)

######################################################
# DEFINE HELPER METHODS

# method to transform mls value to class space
def target_transform(mls):
    return target_transform_partial(mls, time_NaN_fill= int(-10))
    
# method to apply tokenization to the batched texts
def tokenize_function_partial(tokenizer, batched_data, chunk_size):
    result = tokenizer(batched_data['text'], padding = 'max_length', truncation = True, max_length = chunk_size)
    return result

# method to group the texts into chunks for training (not used in final methodology)
def group_texts(batched_data, chunk_size):
    concatenated_examples = {k: sum(batched_data[k], []) for k in batched_data.keys()}
    total_length = len(concatenated_examples[list(batched_data.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {k: [t[i: i+chunk_size] for i in range (0, total_length, chunk_size)] for k, t in concatenated_examples.items()}
    return result


######################################################
# MAIN CODE TO BE EXECUTED BELOW

@validate_call
def train_LLM(
    lookahead_hours: int = 24,
    chunk_size: int = 2560,
    batch_size: int = 10,
    ):

    wandb.init(project = f"final_LLM_{lookahead_hours}hr_finetuning", config = locals(), save_code = True)

    # Get the processed OLTVs, OTVs, and targets using pre_LLM mode on process_data()
    OLTV, OTV, targets, bmc_OLTV, bmc_OTV, bmc_targets = process_data(lookahead_hours = lookahead_hours, pre_LLM = True)

    # Add targets to OLTVs and cut down to only columns needed for LLM training
    OLTV['target'] = targets
    bmc_OLTV['target'] = bmc_targets

    data = OLTV[['ptid', 'rad_text', 'target']]
    bmc_data = bmc_OLTV[['rad_text', 'target']]

    # rename columns to match tokenizer expectation
    data = data.rename(columns = {'rad_text':'text', 'target': 'labels'})
    bmc_data = bmc_data.rename(columns = {'rad_text': 'text', 'target': 'labels'})

    # Split into LLM training and test sets based on ptid (to avoid patient spill over)
    ptids = data.ptid.unique()
    train_ptids, test_ptids = train_test_split(ptids, test_size=0.3)

    train_data = data[data.ptid.isin(train_ptids)].copy()
    test_data = data[data.ptid.isin(test_ptids)].copy()

    # drop ptid column from test and train sets
    train_data.drop(columns = ['ptid'], inplace = True)
    test_data.drop(columns = ['ptid'], inplace = True)

    # Save test set ptids for later testing
    pd.DataFrame(data = test_ptids, columns = ['ptid']).to_json(f"data/processed/test_set_ids/fine_tuning_{lookahead_hours}hr.json")
    
    # convert to dataset object (needed for interface with transformers package)
    train_set = Dataset.from_pandas(train_data, split = "train")
    test_set = Dataset.from_pandas(test_data, split = "test")
    bmc_set = Dataset.from_pandas(bmc_data, split = "test")

    # use gpu (cuda) if available
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer", num_labels = 4).to(device)
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

    def tokenize_function(batched_data):
        return tokenize_function_partial(tokenizer, batched_data, chunk_size)

    # tokenize the text datasets
    tokenized_train_set = train_set.map(tokenize_function, batched = True, batch_size = batch_size)
    tokenized_test_set = test_set.map(tokenize_function, batched = True, batch_size = batch_size)
    tokenized_bmc_set = bmc_set.map(tokenize_function, batched = True, batch_size = batch_size)

    # steps for tracking progress
    logging_steps = len(tokenized_train_set) // batch_size 

    # instantiate evaluator and define function for evaluating progress of fine tuning
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis = -1)
        return metric.compute(predictions = predictions, references = labels)

    # instantiate training arguments 
    training_args = TrainingArguments(
            output_dir = f"models/fine_tuning_cl_{lookahead_hours}hr",
            overwrite_output_dir = True,
            eval_strategy = "epoch",
            num_train_epochs = 6,
            learning_rate = 2e-5,
            weight_decay = 0.01,
            push_to_hub = False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=True,
            logging_steps = logging_steps,
            run_name = f"vm_finetune_{lookahead_hours}hr_for_final"
        )
    
    # instantiate the trainer and run
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train_set,
        eval_dataset = tokenized_test_set,
        compute_metrics = compute_metrics,
    )

    trainer.train()

    # evaluate on mgb test set and bmc 
    mgb_test_results = trainer.evaluate(eval_dataset = tokenized_test_set)
    bmc_eval_results = trainer.evaluate(eval_dataset = tokenized_bmc_set)

    print('MGB TEST RESULTS:')
    print(mgb_test_results)
    wandb.log({"MGB LLM eval results": mgb_test_results})

    print('BMC RESULTS:')
    print(bmc_eval_results)
    wandb.log({"BMC LLM eval results": bmc_eval_results})

    # save fine-tuned model for future use
    trainer.save_model() 
    

######################################################
# CODE EXECUTION

if __name__ == "__main__":
    fire.Fire(train_LLM)