from __future__ import print_function
import numpy as np
import pandas as pd


class TreeNode(object):
    def __init__(self, ids=None, children=[], gini=0, depth=0):
        self.ids = ids  # index of data in this node
        self.gini = gini  # gini, will fill later
        self.depth = depth  # distance to root node
        self.split_attribute = None  # which attribute is chosen, it non-leaf
        self.children = children  # list of its child nodes
        self.order = None  # order of values of split_attribute in children
        self.label = None  # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def gini_index(freq):
    # remove prob 0
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0 / float(freq_0.sum())
    return 1 - np.sum(prob_0**2)


class DecisionTree(object):
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):

        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.Ntrain = 0
        self.min_gain = min_gain

    def _init_data(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data
        self.attributes = list(data)
        self.target = target
        self.labels = target.unique()

    def fit(self, data, target):
        self._init_data(data, target)
        ids = range(self.Ntrain)
        self.root = TreeNode(ids=ids, gini=self._gini_index(ids), depth=0)
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth or node.gini < self.min_gain:
                node.children = self._split(node)
                if not node.children:  # leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)

    def _gini_index(self, ids):
        # calculate gini of a node with index ids
        if len(ids) == 0:
            return 0
        ids = [i + 1 for i in ids]  # panda series index starts from 1
        freq = np.array(self.target[ids].value_counts())
        return gini_index(freq)

    def _set_label(self, node):
        # find label for a node if it is a leaf
        # simply chose by major voting
        target_ids = [i + 1 for i in node.ids]  # target is a series variable
        node.set_label(self.target[target_ids].mode()[0])  # most frequent label

    def _split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()
            if len(values) == 1:
                continue  # gini = 0
            splits = []
            for val in values:
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id - 1 for sub_id in sub_ids])
            # don't split if a node has too small number of points
            if min(map(len, splits)) < self.min_samples_split:
                continue
            # information gain
            HxS = 0
            for split in splits:
                HxS += len(split) * self._gini_index(split) / len(ids)
            gain = node.gini - HxS
            if gain < self.min_gain:
                continue  # stop if small gain
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids=split,
                                gini=self._gini_index(split), depth=node.depth + 1) for split in best_splits]
        return child_nodes

    # def save_model(self):
    #
    #     # travesal from root to leaf bfs

    def predict(self, new_data):
        """
        :param new_data: a new dataframe, each row is a datapoint
        :return: predicted labels for each row
        """
        npoints = new_data.count()[0]
        labels = [None] * npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]  # one point
            # start from root and recursively travel if not meet a leaf
            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label

        return labels


if __name__ == "__main__":
    df = pd.read_csv('adult_train.csv', index_col=0, parse_dates=True)
    # replace >50K: 1 <=50K: 0
    df['income'] = df['income'].replace(['<=50K', '>50K'],
                                        [0, 1])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    tree = DecisionTree(max_depth=3, min_samples_split=2)
    tree.fit(X, y)
    print(tree.predict(X))