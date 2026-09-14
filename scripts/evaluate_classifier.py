import os
import numpy as np
import pandas as pd
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import evaluate
from evaluate import evaluator
import pydantic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, classification_report
import torch
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassAveragePrecision, MulticlassPrecision, MulticlassPrecisionRecallCurve, MulticlassROC
from datasets import Dataset


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


def target_to_label(target):
    if target == 0:
        return "LABEL_0"
    elif target == 1:
        return "LABEL_1"
    elif target == 2:
        return "LABEL_2"
    elif target == 3:
        return "LABEL_3"
    else:
        return "NONE"


def split_test(data, test_ptids):
    return data[data.ptid.isin(test_ptids)].copy()

if __name__ == "__main__":

    OLTV = pd.read_json(os.getcwd() + "/data/processed/OLTV_wshorttext_24hr.json")
    bmc_OLTV = pd.read_json(os.getcwd() + "/data/processed/bmc_OLTV_wshorttext_24hr.json")

    OLTV = OLTV[['ptid', 'dt', 'targets', 'rad_text']]
    bmc_OLTV = bmc_OLTV[['ptid', 'dt', 'targets', 'rad_text']]

    OLTV['label'] = OLTV['targets'].apply(mls_to_target).apply(target_to_label)
    bmc_OLTV['label'] = OLTV['targets'].apply(mls_to_target).apply(target_to_label)

    OLTV.drop('targets', axis = 1, inplace = True)
    bmc_OLTV.drop('targets', axis = 1, inplace = True)

    OLTV.rename(columns = {'rad_text': 'text'}, inplace = True)
    bmc_OLTV.rename(columns = {'rad_text':'text'}, inplace = True)

    test_ptids = list(pd.read_json(os.getcwd() + "/data/processed/test_set_ids/fine_tuning_cl_24hr_shorttext_2.json")['ptid'])
    OLTV_test = split_test(OLTV, test_ptids)
    bmc_OLTV_test = bmc_OLTV.copy()

    OLTV_test.drop(['ptid','dt'], axis = 1, inplace = True)
    bmc_OLTV_test.drop(['ptid','dt'], axis = 1, inplace = True)

    mgb_test_set = Dataset.from_pandas(OLTV_test, split = "test")
    bmc_test_set = Dataset.from_pandas(bmc_OLTV_test, split = "test")

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = AutoModelForSequenceClassification.from_pretrained("models/fine_tuning_cl_24hr_shorttext_2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")

    def tokenize_dataset(dataset):
        result = tokenizer(dataset['text'], padding = 'max_length', truncation = True, max_length = 2560)
        return result
    
    batch_size = 5

    pipe = pipeline("text-classification", model = model, tokenizer = tokenizer)

    #mgb_tokenized_test_set = mgb_test_set.map(tokenize_dataset)
    #mgb_tokenized_test_set = mgb_tokenized_test_set.remove_columns(['text', '__index_level_0__'])

    mgb_test_output = pipe(mgb_test_set['text'], return_all_scores = True)

    bmc_test_output = pipe(bmc_test_set['text'], return_all_scores = True)

    def modify_output(output):
        scores = [[d['score'] for d in D] for D in output]
        output_table = pd.DataFrame(data = scores, columns = ['LABEL_0','LABEL_1', 'LABEL_2', 'LABEL_3'])
        labels = output_table.idxmax(axis = 1)

        return output_table, labels
    
    mgb_test_table, mgb_test_labels = modify_output(mgb_test_output)
    bmc_test_table, bmc_test_labels = modify_output(bmc_test_output)
    
    def get_true_labels(data, output_table):
        labels = list(data['label'])
        true_table = np.zeros(shape = output_table.shape)
        for i in range(len(labels)):
            true_table[i, 0] = 1 if labels[i] == 'LABEL_0' else 0
            true_table[i, 1] = 1 if labels[i] == 'LABEL_1' else 0
            true_table[i, 2] = 1 if labels[i] == 'LABEL_2' else 0
            true_table[i, 3] = 1 if labels[i] == 'LABEL_3' else 0
        
        true_table = pd.DataFrame(data = true_table, columns = ['LABEL_0', 'LABEL_1', 'LABEL_2', 'LABEL_3'])

        return true_table, labels
    
    mgb_true_table, mgb_true_labels = get_true_labels(mgb_test_set, mgb_test_table)
    bmc_true_table, bmc_true_labels = get_true_labels(bmc_test_set, bmc_test_table)

    print('MGB RESULTS:')
    print(f"Accuracy: {accuracy_score(mgb_true_labels, mgb_test_labels)}")
    print(f"Precision: {precision_score(mgb_true_labels, mgb_test_labels, average='weighted')}")
    print(f"AUROC: {roc_auc_score(mgb_true_table, mgb_test_table, average = 'weighted')}")
    print(f"Per-Class Performance: {classification_report(mgb_true_labels, mgb_test_labels)}")

    print('BMC RESULTS:')
    print(f"Accuracy: {accuracy_score(bmc_true_labels, bmc_test_labels)}")
    print(f"Precision: {precision_score(bmc_true_labels, bmc_test_labels, average='weighted')}")
    print(f"AUROC: {roc_auc_score(bmc_true_table, bmc_test_table, average = 'weighted')}")
    print(f"Per-Class Performance: {classification_report(bmc_true_labels, bmc_test_labels)}")














    


