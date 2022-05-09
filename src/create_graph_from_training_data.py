import argparse

import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='file', required=True)
    args = vars(parser.parse_args())
    filname = args['file']

    df = pd.read_csv(filname)

    epochs = df['epoch']
    training_loss = df['Training Loss']
    validation_accuracy = df['Valid. Accur.']
    validation_loss = df['Valid. Loss']

    plt.plot(epochs, training_loss, label='Training loss')
    plt.plot(epochs, validation_loss, label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(epochs, validation_accuracy)
    plt.title('Valid. Accur.')
    plt.xlabel('Epoch')
    plt.ylabel('Valid. Accur.')
    # plt.show()
    plt.savefig('valid_accur.png')
