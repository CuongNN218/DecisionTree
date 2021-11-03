import numpy as np
import pandas as pd


class TreeNode(object):
    def __init__(self, ids=None, children=[], gini=0, depth=0):
        self.ids = ids  # index of data in this node
        self.gini = gini  # gini, will fill later
        self.depth = depth  # distance to root node
        self.split_attribute = None  # which attribute is chosen, it non-leaf
        self.left = None
        self.right = None
        self.order = None  # order of values of split_attribute in children
        self.label = None  # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label

        