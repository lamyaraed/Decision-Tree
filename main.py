import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint as pprint
import random


class Node:
    def __init__(self, data_set, parent=None, child1=None, child2=None, entropy=None, value=None, attr=None,
                 classification=None):
        self.data_set = data_set
        self.parent = parent
        self.child1 = child1
        self.child2 = child2
        self.entropy = entropy
        self.value = value
        self.attr = attr
        self.classification = classification

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

    def set_classification(self, classi):
        self.classification = classi
    # Function to print the tree
    # self is the starting node
    # def BFS(self):
    #
    #     # Mark all the vertices as not visited
    #     visited = [False] * (max(self.graph) + 1)
    #
    #     # Create a queue for BFS
    #     queue = []
    #
    #     # Mark the source node as
    #     # visited and enqueue it
    #     queue.append(self)
    #     visited[self] = True
    #
    #     while queue:
    #
    #         # Dequeue a vertex from
    #         # queue and print it
    #         s = queue.pop(0)
    #         print(s, end=" ")
    #
    #         # Get all adjacent vertices of the
    #         # dequeued vertex s. If a adjacent
    #         # has not been visited, then mark it
    #         # visited and enqueue it
    #         for i in self.graph[s]:
    #             if visited[i] == False:
    #                 queue.append(i)
    #                 visited[i] = True


def train_test_split(df, test_size):
    indices = df.index.tolist()
    training_size = round(len(df) * test_size)
    training_ind = random.sample(population=indices, k=training_size)
    train_df = df.loc[training_ind]
    test_df = df.drop(training_ind)
    return train_df, test_df


def check_duplication(df):
    duplicats_indecis = []
    duplicated = False
    for row in range(df.shape[0]):
        for i in range(row, df.shape[0]):
            for col in range(df.shape[1] - 1):
                if df.loc[i, col] == df.loc[row, col]:
                    duplicated = True
                else:
                    duplicated = False
                    break
            if duplicated:
                if i in duplicats_indecis:
                    continue
                elif df.loc[i, 16] != df.loc[row, 16]:
                    duplicats_indecis.append(i)
                else:
                    continue
    for y in range(len(duplicats_indecis) - 1, -1, -1):
        index = duplicats_indecis[y]
        df = df.drop([index], axis='index')
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
    best_attr = np.argmax(information_gain)
    return best_attr


def subset(s_data, attribute):
    values = np.unique(s_data[attribute])
    subset1 = s_data[s_data[attribute] == values[0]]
    subset2 = s_data[s_data[attribute] == values[1]]
    return subset1, subset2, values[0], values[1]


def build_decision_tree(subset_data, parent=None, Root=None, count=0):
    split_attribute = best_attribute(subset_data)

    if purity_check(subset_data):
        if Root is None:
            root = Node(subset_data, parent, None, None, calculate_entropy(subset_data), classification(subset_data),
                        None, classification(subset_data))
            return root
        # leaf node
        elif parent is not None:
            # leaf_node = Node(subset_data, parent, None, None, calculate_entropy(subset_data), classification(subset_data),
            #                  None, classification(subset_data))
            # parent.set_child1(leaf_node)
            parent.set_entropy(calculate_entropy(subset_data))
            parent.set_classification(classification(subset_data))
            return Root

    # Root node
    elif Root is None:
        child1_data, child2_data, child1_value, child2_value = subset(subset_data, split_attribute)
        root = Node(subset_data, None, None, None, calculate_entropy(subset_data), None, split_attribute, None)
        child1 = Node(child1_data, root, None, None, calculate_entropy(child1_data), child1_value, None, None)
        child2 = Node(child2_data, root, None, None, calculate_entropy(child2_data), child2_value, None, None)
        root.set_child1(child1)
        root.set_child2(child2)
        build_decision_tree(child1_data, child1, root, 1)
        build_decision_tree(child2_data, child2, root, 1)
        return root

    # sub trees
    else:
        child1_data, child2_data, child1_value, child2_value = subset(subset_data, split_attribute)
        child1 = Node(child1_data, parent, None, None, calculate_entropy(child1_data), child1_value, None, None)
        child2 = Node(child2_data, parent, None, None, calculate_entropy(child2_data), child2_value, None, None)
        parent.set_child1(child1)
        parent.set_child2(child2)
        parent.set_attr(split_attribute)

        build_decision_tree(child1_data, child1, Root, count + 1)
        build_decision_tree(child2_data, child2, Root, count + 1)
        return Root


def predictData(test_row, decision_tree, i):
    if decision_tree.child1 is not None:
        current_attr = decision_tree.attr
        child1_node = decision_tree.child1
        child2_node = decision_tree.child2
        child1_value = child1_node.value
        test_cell = test_row.loc[i, current_attr]
        if test_cell == child1_value:
            return predictData(test_row, child1_node, i)
        else:
            return predictData(test_row, child2_node, i)

    else:
        return decision_tree.classification


def tree_size(root, count):
    if root.child1 is not None:
        count = tree_size(root.child1, count+1)
        count = tree_size(root.child2, count+1)
        return count
    else:
        return count


def main(training_size):
    dataset = pd.read_csv('dataSet.csv', header=None)
    accuracy_list = []
    trees_sizes = []
    for i in range(0, 5):
        train_dataset, test_dataset = train_test_split(dataset, test_size=training_size)
        train_dataset.reset_index(inplace=True, drop=True)
        test_dataset.reset_index(inplace=True, drop=True)

        train_dataset = replace(train_dataset)
        test_dataset = replace(test_dataset)

        train_dataset.reset_index(inplace=True, drop=True)
        train_dataset = check_duplication(train_dataset)
        train_dataset.reset_index(inplace=True, drop=True)

        # print("train")
        # print(train_dataset)
        node = build_decision_tree(train_dataset)
        s_tree = tree_size(node, 1)
        correct_predections = 0
        for i in range(test_dataset.shape[0]):
            decision = predictData(test_dataset.loc[[i]], node, i)
            if decision == test_dataset.loc[i, 16]:
                correct_predections += 1
        accuracy = (correct_predections/len(test_dataset) * 100)
        accuracy_list.append(accuracy)
        trees_sizes.append(s_tree)

    max_accuracy = np.max(accuracy_list)
    max_tree_size = np.max(trees_sizes)

    min_accuracy = np.min(accuracy_list)
    min_tree_size = np.min(trees_sizes)

    mean_accuracies = np.mean(accuracy_list)
    mean_tree_sizes = np.mean(trees_sizes)

    print("For training size ", training_size, "  :")
    print("")
    print("Max accuracy :", max_accuracy)
    print("Min accuracy :", min_accuracy)
    print("Mean accuracy :", mean_accuracies)
    print("")
    print("Max tree size :", max_tree_size)
    print("Min tree size :", min_tree_size)
    print("Mean tree size :", mean_tree_sizes)
    print("\n-------------------------------------------------------\n")

for i in range(30, 80, 10):
    size = i/100
    main(size)

# main(0.30)
# print("\n \n")
# print(train_dataset.describe())
# print(train_dataset.loc[1, 1])
# print(train_dataset.shape[0])  # rows
# print(train_dataset.shape[1])  # cols
# print(train_dataset.describe().loc['top', 0])
# node = Node(train_dataset, None, None, None, 15, None, None)

# print(node)
# print(node.entropy)
# print(node.attr)
# print(node.child1.value)
# print(node.child2.value)
# print(node.child1.attr)
# print(node.child1.child1.value)
# print(node.child1.child2.value)
# print(train_dataset[train_dataset[0] == 'n'])
# print("subset col 0 \n ")
# print(subset(train_dataset, 0))
# print("subset col 16 \n ")
# print(subset(train_dataset, 16))
