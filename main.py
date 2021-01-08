import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as pprint
import random


# todo ? ll top                              ---------DONE
# todo entropy ll parent =                   ---------DONE
# todo entropies ll children                 ---------DONE
# todo inf gain 3la asasha a split
# todo after splitting, ne7seb entropy ll


class Node:
    def __init__(self, data_set, parent=None, child1=None, child2=None, entropy=None, value=None):
        self.data_set = data_set
        self.parent = parent
        self.child1 = child1
        self.child2 = child2
        self.entropy = entropy
        self.value = value

    def set_parent(self, parent):
        self.parent = parent

    def set_child1(self, child1):
        self.child1 = child1

    def set_child2(self, child2):
        self.child2 = child2

    def set_entropy(self, entropy):
        self.entropy = entropy

    def set_value(self, value):
        self.value = value


def train_test_split(df, test_size):
    indices = df.index.tolist()
    training_size = round(len(df) * test_size)
    training_ind = random.sample(population=indices, k=training_size)
    train_df = df.loc[training_ind]
    test_df = df.drop(training_ind)
    return train_df, test_df


def replace(rData):
    description = rData.describe()
    for i in range(rData.shape[1] - 1):
        top = description.loc['top', i]
        for y in range(rData.shape[0]):
            if rData.loc[y, i] == '?':
                rData.loc[y, i] = top

    return rData


def purity_check(test_data):
    label = test_data[:, -1]
    unique = np.unique(label)
    if len(unique) == 1:
        return True
    else:
        return False


def classification(test_data):
    label = test_data[:, -1]
    unique_classes, count_classes = np.unique(label, return_counts=True)
    classified = unique_classes[count_classes.argmax()]
    return classified


def calculate_entropy(ndata):
    classes, class_count = np.unique(ndata, return_counts=True)
    entropy_value = np.sum([(-classes[i] / np.sum(class_count)) * np.log2(class_count[i] / np.sum(class_count)) for i in
                            range(len(classes))])
    return entropy_value


def calculate_information_gain(ndata, label, index):
    # calculate parent entropy
    parent_entropy = calculate_entropy(ndata[label])
    values, features_counts = np.unique(data[index], return_counts=True)

    # information gain
    features_entropy = np.sum([(features_counts[i]/np.sum(features_counts))* calculate_entropy(ndata.where(ndata[index] == values[i]).dropna()[label])for i in range(len(values))])
    feature_information_gain = parent_entropy - features_entropy
    return feature_information_gain


def Decision_Tree(treeData):
    # if purity_check(treeData):
    #     leaf_node = Node(treeData, prevNode, None, None, entropy, classification(treeData))
    information_gain = []
    for i in range(0, 15):
        information_gain.append(calculate_information_gain(treeData, treeData[treeData.shape[1]-1], i))
    print("information gain")
    print(information_gain)


dataset = pd.read_csv('dataSet.csv', header=None)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.25)
train_dataset.reset_index(inplace=True, drop=True)
test_dataset.reset_index(inplace=True, drop=True)
Decision_Tree(train_dataset)

# print("train")
# print(train_dataset)
# print("test")
# print(test_dataset)

train_dataset = replace(train_dataset)
test_dataset = replace(test_dataset)

data = train_dataset.values

# if purity_check(data):
#     classification(data)
# else:
#     split = to_split(data)
print("train")
print(train_dataset)
print("test")
print(test_dataset)
print("\n \n")
print(train_dataset.describe())
print(train_dataset.loc[1, 1])
print(train_dataset.shape[0])  # rows
print(train_dataset.shape[1])  # cols
print(train_dataset.describe().loc['top', 0])
