import glob

import pandas as pd
from matplotlib import pyplot as plt

model_name = 'bert-large-uncased'

folder = '/Users/gabriel-he/Documents/n2c2 shared task/models/kfold_bert-large-uncased'
files = glob.glob(folder + '/**/training_status.csv', recursive=True)
files = sorted(files)

df = []
for file in files:
    df.append(pd.read_csv(file))

plt.figure()
plt.title('Training Loss ({})'.format(model_name))
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
for i in range(len(df)):
    epochs = df[i]['epoch']
    training_loss = df[i]['Training Loss']
    plt.plot(epochs, training_loss, label='iter ' + str(i))
plt.legend()
# plt.show()
plt.savefig('training_loss.png')
plt.yscale('log')
plt.savefig('training_loss_log.png')

plt.figure()
plt.title('Valid. Accur. ({})'.format(model_name))
plt.xlabel('Epoch')
plt.ylabel('Valid. Accur.')
for i in range(len(df)):
    epochs = df[i]['epoch']
    validation_accuracy = df[i]['Valid. Accur.']
    plt.plot(epochs, validation_accuracy, label='iter ' + str(i))
plt.legend()
# plt.show()
plt.savefig('valid_accur.png')

plt.figure()
plt.title('Valid. Loss ({})'.format(model_name))
plt.xlabel('Epoch')
plt.ylabel('Valid. Loss')
for i in range(len(df)):
    epochs = df[i]['epoch']
    validation_loss = df[i]['Valid. Loss']
    plt.plot(epochs, validation_loss, label='iter ' + str(i))
plt.legend()
# plt.show()
plt.savefig('valid_loss.png')
