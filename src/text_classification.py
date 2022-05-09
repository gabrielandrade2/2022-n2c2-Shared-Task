import csv

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


def numeric_category(train_category, test_category):
    categories_df = pd.DataFrame(categories, columns=['category'])
    categories_df['labels'] = le.fit_transform(categories_df['category'])
    le.fit(train_category)
    le.fit(test_category)
    train_labels = le.transform(train_category)
    test_labels = le.transform(test_category)
    return train_labels, test_labels, categories_df


"""Define functions to calculate the accuracy of predictions vs true labels and elapsed time as hh:mm:ss"""
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_model(output_dir):
    print("Saving model to %s" % output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(output_dir + 'training_status.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, training_status[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(training_status)

    with open(output_dir + 'outfile.csv', 'w') as outfile:
        outfile.write("text,predict_label,true_label,predict_probability")
        for s, p, prob, t in zip(sentences, result_label, result_prob, true_labels_array):
            outfile.write('"{}",{},{},{}\n'.format(''.join(s), categories[p], categories[t], prob))


def create_graphs(output_dir):
    epochs = [t['epoch'] for t in training_status]
    training_loss = [t['Training Loss'] for t in training_status]
    validation_accuracy = [t['Valid. Accur.'] for t in training_status]
    validation_loss = [t['Valid. Loss'] for t in training_status]

    plt.plot(epochs, training_loss)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.show()
    plt.savefig(output_dir + 'training_loss.png')

    plt.figure()
    plt.plot(epochs, validation_accuracy)
    plt.title('Valid. Accur.')
    plt.xlabel('Epoch')
    plt.ylabel('Valid. Accur.')
    plt.show()
    plt.savefig(output_dir + 'valid_accur.png')

    plt.figure()
    plt.plot(epochs, validation_loss)
    plt.title('Valid. Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Valid. Loss')
    plt.show()
    plt.savefig(output_dir + 'valid_loss.png')

if __name__ == '__main__':

    # Check the available GPU and use it if it is exist. Otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print('GPU:', torch.cuda.get_device_name(1))
    else:
        device = torch.device("cpu")
        print('CPU')

    # ----------Data-------------
    train_file = ('../300-window_split_train_p1.csv')
    test_file = ('../300-window_split_dev_p1.csv')

    print("---Read data---")
    # -----------------train file-------------------------
    train_data = pd.read_csv(train_file, names=["text", "category", "file", "type", "word", "start", "end"])

    train_text = train_data.text.values
    # Convert to lower case
    train_text = [text.lower() for text in train_text]
    train_category = train_data.category.values

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

    train_labels, test_labels, categories_df = numeric_category(train_category, test_category)

    print("Convert non-numeric labels to numeric labels\n")
    print(categories_df.sort_values(by='category', ascending=True))

    """# Text Tokenization
    
    Text Tokenization with BERT tokenizer. BERT model add special tokens [CLS] and [SEP] to the begin and end of sequence.
    Specify the maximum sequence length of the dataset to pad or truncate the sequence.
    
    """

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Specify maximum sequence length to pad or truncate
    max_len = 0

    for seq in train_text:
        # Tokenize the text by BERT tokenizer
        input_ids = tokenizer.encode(seq, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    print('Max sequence length    ', max_len)

    # Tokenize all sequences and map the tokens to their IDs.
    input_ids_train = []
    attention_masks_train = []

    # For every sequences
    for seq in tqdm(train_text):
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
        input_ids_train.append(encoded_dict['input_ids'])

        # And its attention mask
        attention_masks_train.append(encoded_dict['attention_mask'])

    input_ids_train = torch.cat(input_ids_train, dim=0)
    attention_masks_train = torch.cat(attention_masks_train, dim=0)
    train_labels = torch.tensor(train_labels)

    """Split train data to use 90% as train and 10% as validation set in the model training phase
    """

    # Change to TensorDataset and Split to train and validation sets (90-10)
    dataset = TensorDataset(input_ids_train, attention_masks_train, train_labels)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('training set', format(train_size))
    print('validation set', format(val_size))

    """#Classification Model
    
    Specify batch size. In fine-tuning BERT model the recommended batch size is 16 or 32.
    """

    # specify batch size
    batch_size = 32

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

    # Specify Classification model

    model = BertForSequenceClassification.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT",
        num_labels=len(categories),
        output_attentions=False,
        output_hidden_states=False,
        ignore_mismatched_sizes=True
    )

    # Run the model on GPU
    if torch.cuda.is_available():
        model.cuda(device=1)

    # Specify the optimizer and epoch number
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    epochs = 100  # recomende 2-4 by BERT model's authors
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    """#Train the Model"""

    # Traning start
    # seed_val = 42
    # random.seed(seed_val)
    # np.random.seed(seed_val)
    # torch.manual_seed(seed_val)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed_val)

    # use training_status to store loss values, accuracy and elapsed time
    training_status = []
    total_t0 = time.time()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_dir = './model_save/' + timestr + '/'
    checkpoint_dir = output_dir + 'checkpoint/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch_i in range(0, epochs):

        # -------------------Training-----------------------
        print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))

        t0 = time.time()
        total_train_loss = 0

        model.train()

        for step, batch in tqdm(enumerate(train_dataloader)):
            if step % 200 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            output = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           labels=b_labels)

            total_train_loss += output.loss.item()
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("\n")
        print(" Average training loss: {0:.4f}".format(avg_train_loss))
        print(" Training epoch took: {:}".format(training_time))

        # ------------------Validation--------------------
        print("\n")
        print("Validation")

        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                output = model(b_input_ids,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels)

            total_eval_loss += output.loss.item()

            output.logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(output.logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_status.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        if epoch_i % 10 == 0:
            output_path = checkpoint_dir + 'checkpoint{}.model'.format(epoch_i)
            torch.save(model.state_dict(), output_path)

    print("\n")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    """#Test the Model"""

    # Tokenize all sequences and map the tokens to their IDs.
    input_ids_test = []
    attention_masks_test = []

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
    print("F1           ", f1_score(test_labels, result_label, average="macro"))
    print(classification_report(true_labels_array, result_label, target_names=target_names))

    """Save the Model"""
    save_model(output_dir)

    """Create graphs"""
    create_graphs(output_dir)
