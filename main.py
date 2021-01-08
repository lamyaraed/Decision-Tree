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
    def __init__(self, data_set, parent=None, child1=None, child2=None, entropy=None, value=None, attr=None):
        self.data_set = data_set
        self.parent = parent
        self.child1 = child1
        self.child2 = child2
        self.entropy = entropy
        self.value = value
        self.attr = attr

    def __repr__(self):
        return ('Node(data_set=%s parent=%s child1=%s child2=%s entropy=%s value=%s attr=%s)'
                % (repr(self.data_set), repr(self.parent), repr(self.child1), repr(self.child2), repr(self.entropy),
                   repr(self.value), repr(self.attr)))

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

    def set_attr(self, attr):
        self.attr = attr


def train_test_split(df, test_size):
    indices = df.index.tolist()
    training_size = round(len(df) * test_size)
    training_ind = random.sample(population=indices, k=training_size)
    train_df = df.loc[training_ind]
    test_df = df.drop(training_ind)
    return train_df, test_df


def check_duplication(df_):
    df = df_
    duplicated = False
    for row in range(df.shape[0]):
        for i in range(df.shape[0]):
            for col in range(df.shape[1]-1):
                if df.loc[i, col] == df.loc[row, col]:
                    duplicated = True
                else:
                    duplicated = False
                    break
            if duplicated:
                if df.loc[i, 16] != df.loc[row, 16]:
                    df.drop(i)
                    print("here")
                else:
                    continue
    return df


def replace(rData):
    description = rData.describe()
    for i in range(rData.shape[1] - 1):
        top = description.loc['top', i]
        for y in range(rData.shape[0]):
            if rData.loc[y, i] == '?':
                rData.loc[y, i] = top

    return rData


# check if last column is pure or not
def purity_check(test_data):
    label = test_data[16]
    unique = np.unique(label)
    if len(unique) == 1:
        return True
    else:
        return False


def classification(test_data):
    label = test_data[16]
    unique_classes, count_classes = np.unique(label, return_counts=True)
    classified = unique_classes[count_classes.argmax()]
    return classified


def calculate_entropy(ndata):
    classes, class_count = np.unique(ndata, return_counts=True)
    entropy_value = np.sum(
        [(-class_count[i] / np.sum(class_count)) * np.log2(class_count[i] / np.sum(class_count)) for i in
         range(0, len(classes))])
    return entropy_value


def calculate_information_gain(ndata, label, index):
    # calculate parent entropy
    parent_entropy = calculate_entropy(ndata[label])
    values, features_counts = np.unique(ndata[index], return_counts=True)

    # information gain
    features_entropy = np.sum([(features_counts[i] / np.sum(features_counts)) * calculate_entropy(
        ndata.where(ndata[index] == values[i]).dropna()[label]) for i in range(len(values))])
    feature_information_gain = parent_entropy - features_entropy
    return feature_information_gain


def best_attribute(subset_data):
    information_gain = []
    for i in range(0, subset_data.shape[1] - 1):
        information_gain.append(calculate_information_gain(subset_data, 16, i))
    print("info gain")
    print(information_gain)
    best_attr = np.argmax(information_gain)
    return best_attr


def subset(s_data, attribute):
    values = np.unique(s_data[attribute])
    print("attribute")
    print(attribute)
    print("subset values")
    print(values)
    print("data set")
    print(s_data)
    subset1 = s_data[s_data[attribute] == values[0]]
    subset2 = s_data[s_data[attribute] == values[1]]
    return subset1, subset2, values[0], values[1]


def build_decision_tree(subset_data, parent, Root, count=None):
    split_attribute = best_attribute(subset_data)

    # leaf node
    if purity_check(subset_data):
        leaf_node = Node(subset_data, parent, None, None, calculate_entropy(subset_data), classification(subset_data),
                         None)
        parent.set_child1(leaf_node)
        return Root

    # Root node
    elif parent is None:
        child1_data, child2_data, child1_value, child2_value = subset(subset_data, split_attribute)
        root = Node(subset_data, None, None, None, calculate_entropy(subset_data), None, None)
        child1 = Node(child1_data, root, None, None, calculate_entropy(child1_data), child1_value, split_attribute)
        child2 = Node(child2_data, root, None, None, calculate_entropy(child2_data), child2_value, split_attribute)
        root.set_child1(child1)
        root.set_child2(child2)
        # print(" root data set")
        # print(root.data_set)
        # print(" root split attribute")
        # print(split_attribute)
        # print(" root parent")
        # print(root.parent)
        # print(" root entropy")
        # print(root.entropy)
        # print(" root child1 value")
        # print(root.child1.value)
        # print(" root child2 value")
        # print(root.child2.value)
        build_decision_tree(child1_data, child1, root, 1)
        build_decision_tree(child2_data, child2, root, 1)
        # return Node(Root.data_set, Root.parent, Root.child1, Root.child2, Root.entropy, Root.value, Root.attr)

    # sub trees
    else:
        child1_data, child2_data, child1_value, child2_value = subset(subset_data, split_attribute)
        child1 = Node(child1_data, parent, None, None, calculate_entropy(child1_data), child1_value, split_attribute)
        child2 = Node(child2_data, parent, None, None, calculate_entropy(child2_data), child2_value, split_attribute)
        parent.set_child1(child1)
        parent.set_child2(child2)
        # print("parent data set")
        # print(count)
        # print(parent.data_set)
        # print(" parent split attribute")
        # print(split_attribute)
        # print(" parent parent")
        # print(parent.parent.value)
        # print(" parent entropy")
        # print(parent.entropy)
        # print(" parent child1 value")
        # print(parent.child1.value)
        # print(" parent child2 value")
        # print(parent.child2.value)

        build_decision_tree(child1_data, child1, Root, count + 1)
        build_decision_tree(child2_data, child2, Root, count + 1)
        return Root


dataset = pd.read_csv('dataSet.csv', header=None)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.25)
train_dataset.reset_index(inplace=True, drop=True)
test_dataset.reset_index(inplace=True, drop=True)
# build_decision_tree(train_dataset)

# print("train")
# print(train_dataset)
# print("test")
# print(test_dataset)

train_dataset = replace(train_dataset)
test_dataset = replace(test_dataset)

train_dataset = check_duplication(train_dataset)
test_dataset = check_duplication(test_dataset)

train_dataset.reset_index(inplace=True, drop=True)
test_dataset.reset_index(inplace=True, drop=True)

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
# print(train_dataset.describe().loc['top', 0])
# node = Node(train_dataset, None, None, None, 15, None, None)
node = build_decision_tree(train_dataset, None, None)
# print(node)
# print(node.entropy)

# print(train_dataset[train_dataset[0] == 'n'])
# print("subset col 0 \n ")
# print(subset(train_dataset, 0))
# print("subset col 16 \n ")
# print(subset(train_dataset, 16))
