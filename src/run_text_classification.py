import csv
import glob

import pandas as pd
import numpy as np
import time
import os
import random
import datetime
import operator
import re
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (BertForSequenceClassification,
                          BertTokenizer,
                          get_linear_schedule_with_warmup,
                          )
from sklearn import preprocessing
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt


def normalize_text(text):
    # do lowercase
    result_lower = ''.join((text.lower()))
    # remove numbers
    result_non_numeric = ''.join([i for i in result_lower if not i.isdigit()])
    # remove punctuations
    result_non_punc = re.sub(r"[*'`,”+;,_,’^#@<=>~&$€%[:!\-\"\\\/}{?\,.()[\]{|}]", '', result_non_numeric).strip()
    # remove whitespace
    result_non_space = re.sub(' +', ' ', result_non_punc)

    return (result_non_space)


def numeric_category(test_category):
    categories_df = pd.DataFrame(categories, columns=['category'])
    categories_df['labels'] = le.fit_transform(categories_df['category'])
    le.fit(test_category)
    test_labels = le.transform(test_category)
    return test_labels, categories_df


"""Define functions to calculate the accuracy of predictions vs true labels and elapsed time as hh:mm:ss"""


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == '__main__':

    # Check the available GPU and use it if it is exist. Otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print('GPU:', torch.cuda.get_device_name(1))
    else:
        device = torch.device("cpu")
        print('CPU')

    # ----------Data-------------
    test_file = ('../200-window_split_dev_p1.csv')

    print("---Read data---")
    # -----------------test file-------------------------
    test_data = pd.read_csv(test_file, names=["text", "category", "file", "type", "word", "start", "end"])

    test_text = test_data.text.values
    # Convert to lower case
    test_text = [text.lower() for text in test_text]
    test_category = test_data.category.values

    print("---Complete reading data---")

    # Convert non-numeric labels to numeric labels.
    categories = ('Disposition', 'NoDisposition', 'Undetermined')
    le = preprocessing.LabelEncoder()

    test_labels, categories_df = numeric_category(test_category)

    print("Convert non-numeric labels to numeric labels\n")
    print(categories_df.sort_values(by='category', ascending=True))

    """# Text Tokenization
    
    Text Tokenization with BERT tokenizer. BERT model add special tokens [CLS] and [SEP] to the begin and end of sequence.
    Specify the maximum sequence length of the dataset to pad or truncate the sequence.
    
    """
    for model_path in glob.glob('./kfold_bert-large-uncased/*/'):

        print(model_path)


        # ----------Model-------------
        print("---Read model---")

        # model_path = './model_save/20220426-174407/'

        # Load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path)

        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(categories),
            output_attentions=False,
            output_hidden_states=False,
            ignore_mismatched_sizes=True
        )
        # model.load_state_dict(
        #     torch.load('/Users/gabriel-he/PycharmProjects/n2c2-task/src/20220426-174407/pytorch_model.bin',
        #                map_location=device))

        # Run the model on GPU
        if torch.cuda.is_available():
            model.cuda(device=1)

        print("---Complete reading model---")

        """#Test the Model"""

        # Tokenize all sequences and map the tokens to their IDs.
        input_ids_test = []
        attention_masks_test = []

        # Specify maximum sequence length to pad or truncate
        max_len = 0

        for seq in test_text:
            # Tokenize the text by BERT tokenizer
            input_ids = tokenizer.encode(seq, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))

        # For every sequences
        sentences = []
        for seq in test_text:
            sentences.append(seq)
            encoded_dict = tokenizer.encode_plus(
                seq,  # Sequence to encode
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=max_len,
                padding='max_length',  # Pad and truncate
                truncation=True,  # Truncate the seq
                return_attention_mask=True,  # Construct attn. masks
                return_tensors='pt',  # Return pytorch tensors
            )

            # Add the encoded sequences to the list
            input_ids_test.append(encoded_dict['input_ids'])

            # And its attention mask
            attention_masks_test.append(encoded_dict['attention_mask'])

        input_ids_test = torch.cat(input_ids_test, dim=0)
        attention_masks_test = torch.cat(attention_masks_test, dim=0)
        test_labels = torch.tensor(test_labels)

        batch_size = 32

        prediction_data = TensorDataset(input_ids_test, attention_masks_test, test_labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # Prediction on test set

        print('Predicting labels for {:,} test sentences'.format(len(input_ids_test)))

        model.eval()
        predictions, true_labels = [], []

        for batch in tqdm(prediction_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # [sentences.append(tokenizer.convert_ids_to_tokens(t)) for t in b_input_ids]
            predictions.append(logits)
            true_labels.append(label_ids)

        """Evaluate the Model"""

        true_labels_array, result_label, result_prob, result_logits = [], [], [], []
        for j in range(len(true_labels)):
            for i in range(len(true_labels[j])):
                true_labels_array.append(true_labels[j][i])

        for j in range(len(predictions)):
            for i in range(len(predictions[j])):
                index, value = max(enumerate(predictions[j][i]), key=operator.itemgetter(1))
                result_label.append(index)
                result_prob.append(value)
                result_logits.append(predictions[j][i])

        target_names = categories

        print("Accuracy     ", accuracy_score(test_labels, result_label))
        print("Precision    ", precision_score(test_labels, result_label, average="macro"))
        print("Recall       ", recall_score(test_labels, result_label, average='macro'))
        print("F1 macro     ", f1_score(test_labels, result_label, average="macro"))
        print("F1 micro     ", f1_score(test_labels, result_label, average="micro"))
        print(classification_report(true_labels_array, result_label, target_names=target_names))

        classification = [categories[i] for i in result_label]

        # my forecast file df has columns:  sentence	true_label	pred_label	drug	span_start	span_end	anno_type	filename
        df = pd.DataFrame(list(zip(
             test_data['text'].to_list(), test_data['category'].to_list(), classification, test_data['word'].to_list(),
             test_data['start'].to_list(), test_data['end'].to_list(), test_data['type'].to_list(),
             test_data['file'].to_list())),
            columns=['sentence', 'true_label', 'pred_label', 'drug', 'span_start', 'span_end', 'anno_type', 'filename'])

        # df.to_csv('prediction_results_300_1.csv')
