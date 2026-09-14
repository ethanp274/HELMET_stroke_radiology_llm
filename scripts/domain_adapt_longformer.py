import numpy as np
import pandas as pd
import transformers
import os
import sys
import datasets
from datasets import Dataset
from pydantic import confloat, validate_arguments
from typing import Literal
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
import datetime
from datetime import timedelta
import json
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassAveragePrecision
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from tqdm.auto import tqdm
import evaluate

from helmet_mass_effect_pred.data.external_cohort import create_MGH_and_BMC
from helmet_mass_effect_llm.text_generation import (
    create_OLTVs, 
    get_cleaned_data,
    generate_labels,
    data_to_text,
    make_dataset,
    get_sets_from_OLTV
)




if __name__ == "__main__":
    
    train_OLTV, train_targets, test_OLTV, test_targets, bmc_OLTV, bmc_targets = get_cleaned_data()

    train_set, test_set, bmc_set = get_sets_from_OLTV(train_OLTV, train_targets, test_OLTV, test_targets, bmc_OLTV, bmc_targets)
    
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer").to(device)
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

    chunk_size = 1024
    batch_size = 4
    
    def tokenize_function(batched_data):
        result = tokenizer(batched_data['text'], padding = 'max_length', truncation = True, max_length = 1024)
        if tokenizer.is_fast:
            result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
        return result
    
    def group_texts(batched_data):
        concatenated_examples = {k: sum(batched_data[k], []) for k in batched_data.keys()}
        total_length = len(concatenated_examples[list(batched_data.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size
        result = {k: [t[i: i+chunk_size] for i in range (0, total_length, chunk_size)] for k, t in concatenated_examples.items()}
        result['labels'] = result['input_ids'].copy()
        return result

    tokenized_train_set = train_set.map(tokenize_function, batched = True, remove_columns = ['text', 'labels'])
    tokenized_test_set = test_set.map(tokenize_function, batched = True, remove_columns = ['text', 'labels'])

    lm_train_set = tokenized_train_set.map(group_texts, batched = True)
    lm_test_set = tokenized_test_set.map(group_texts, batched = True)

    
    logging_steps = len(lm_train_set) // batch_size 
    
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm_probability = 0.15)

    training_args = TrainingArguments(
        output_dir = "models/fine_tuning_cl", 
        overwrite_output_dir = True,
        evaluation_strategy = "epoch",
        num_train_epochs = 7,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        push_to_hub = False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        logging_steps = logging_steps,
        run_name = "vm_cl_finetune1"
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = lm_train_set,
        eval_dataset = lm_test_set,
        data_collator = data_collator
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Loss: {eval_results['eval_loss']:.3f}")

    trainer.save_model()

