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


def mls_to_target(mls):
    if mls == -10:
        return -1
    elif mls == 0:
        return 0
    elif mls < 3:
        return 1
    elif mls < 8:
        return 2
    else:
        return 3


if __name__ == "__main__":
    
    OLTV = pd.read_json(os.getcwd() + "/data/processed/OLTV_wshorttext_24hr.json")
    bmc_OLTV = pd.read_json(os.getcwd() + "/data/processed/bmc_OLTV_wshorttext_24hr.json")

    OLTV = OLTV[['ptid', 'targets', 'rad_text']]
    bmc_OLTV = bmc_OLTV[['ptid', 'targets', 'rad_text']]

    OLTV['label'] = OLTV['targets'].apply(mls_to_target)
    bmc_OLTV['label'] = OLTV['targets'].apply(mls_to_target)

    OLTV.drop('targets', axis = 1, inplace = True)
    bmc_OLTV.drop('targets', axis = 1, inplace = True)

    OLTV.rename(columns = {'rad_text': 'text'}, inplace = True)
    bmc_OLTV.rename(columns = {'rad_text':'text'}, inplace = True)

    ptids = OLTV.ptid.unique()
    train_ptids, test_ptids = train_test_split(ptids, test_size=0.3)
    pd.DataFrame(data = test_ptids, columns = ['ptid']).to_json(os.getcwd() + "/data/processed/test_set_ids/fine_tuning_cl_24hr_shorttext_2.json")
    train_OLTV = OLTV[OLTV.ptid.isin(train_ptids)].copy()
    test_OLTV = OLTV[OLTV.ptid.isin(test_ptids)].copy()

    train_OLTV.drop('ptid', axis = 1, inplace = True)
    test_OLTV.drop('ptid', axis = 1, inplace = True)
    bmc_OLTV.drop('ptid', axis = 1, inplace = True)

    train_set = Dataset.from_pandas(train_OLTV, split = "train")
    test_set = Dataset.from_pandas(test_OLTV, split = "test")
    bmc_set = Dataset.from_pandas(bmc_OLTV, split = "test")

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AutoModelForSequenceClassification.from_pretrained("yikuan8/Clinical-Longformer", num_labels = 4).to(device)
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

    chunk_size = 2560
    batch_size = 10
    
    def tokenize_function(batched_data):
        result = tokenizer(batched_data['text'], padding = 'max_length', truncation = True, max_length = 2560)
        return result
    
    def group_texts(batched_data):
        concatenated_examples = {k: sum(batched_data[k], []) for k in batched_data.keys()}
        total_length = len(concatenated_examples[list(batched_data.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size
        result = {k: [t[i: i+chunk_size] for i in range (0, total_length, chunk_size)] for k, t in concatenated_examples.items()}
        return result

    tokenized_train_set = train_set.map(tokenize_function, batched = True, batch_size = batch_size)
    tokenized_test_set = test_set.map(tokenize_function, batched = True, batch_size = batch_size)
    tokenized_bmc_set = test_set.map(tokenize_function, batched = True, batch_size = batch_size)

    #lm_train_set = tokenized_train_set.map(group_texts, batched = True)
    #lm_test_set = tokenized_test_set.map(group_texts, batched = True)

    logging_steps = len(tokenized_train_set) // batch_size 

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis = -1)
        return metric.compute(predictions = predictions, references = labels)

    training_args = TrainingArguments(
        output_dir = "models/fine_tuning_cl_24hr_shorttext_2", 
        overwrite_output_dir = True,
        evaluation_strategy = "epoch",
        num_train_epochs = 6,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        push_to_hub = False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        logging_steps = logging_steps,
        run_name = "vm_cl_finetune_raw_text_24hr_shorttext_2"
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train_set,
        eval_dataset = tokenized_test_set,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    bmc_eval_results = trainer.evaluate(eval_dataset = tokenized_bmc_set)

    print('BMC RESULTS:')
    print(bmc_eval_results)

    trainer.save_model()

