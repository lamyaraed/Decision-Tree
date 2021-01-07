import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as pprint
import random


def train_test_split(df, test_size):
    indices = df.index.tolist()
    training_size = round(len(df) * test_size)
    training_ind = random.sample(population=indices, k=training_size)
    train_df = df.loc[training_ind]
    test_df = df.drop(training_ind)
    return train_df, test_df


dataset = pd.read_csv('house-votes-84.data.txt')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.25)


#
# msk = np.random.rand(len(dataset)) < 0.3
# train = dataset[msk]
# test = dataset[~msk]


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
