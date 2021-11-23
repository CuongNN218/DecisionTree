import numpy as np
import pandas as pd
from itertools import combinations
import csv
import argparse

def mapper(df, mapper_dict):
    df = df.replace(mapper_dict)
    return df


# class tree node for create a new node in tree
class Node(object):
    def __init__(self):
        self.left = None  # left child
        self.right = None  # right child

        self.parent_id = None  # id of parents, none for root node
        self.id = 0
        self.splitting_att = None  # splitting attribute
        self.g_left = []  # group of elements for left child
        self.g_right = []  # group of elements for right child
        self.label = -1  # label of the node 0: neg, 1: pos, -1: non-determine
        self.used_atts = None
        # self.n_pos = 0  # number of positive labels for left child
        # self.n_neg = 0  # number of negative labels for right child


class BinaryDT(object):

    def __init__(self, min_sup, data_train, numNode=0):
        self.root = None
        self.min_sup = min_sup
        self.numNode = 0
        self.dataTrain = data_train

    def order2category(self, split_list, attribute):

        if attribute == 'age':

            mapper = {"0": 'young', "1": 'adult', "2": 'senior', "3": 'old'}
            new_split_list = [mapper[item] for item in split_list]

        elif attribute == 'education':
            print(split_list)
            mapper = {"0": 'BeforeHS', "1": 'HS-grad', "2": 'AfterHS', "3": 'Bachelors', "4": 'Grd'}
            new_split_list = [mapper[item] for item in split_list]

        elif attribute == 'hr_per_week':

            mapper = {"0": 'part-time', "1": 'full-time', "2": 'over-time'}
            new_split_list = [mapper[item] for item in split_list]

        else:
            new_split_list = split_list
        return new_split_list

    def gini_index(self, df):
        # get counting number for df
        count_0, count_1 = self.counting_label(df)
        # calculate calculate the gini_index for each node
        if count_0 == 0 or count_1 == 0:
            return 0
        total = count_0 + count_1
        gini_index = 1 - (count_0 / total) ** 2 - (count_1 / total) ** 2
        return gini_index

    def counting_label(self, df):
        counting = df['income'].value_counts().to_dict()
        if '0' not in counting.keys():
            counting['0'] = 0
        if '1' not in counting.keys():
            counting['1'] = 0
        #         print(counting)
        return counting['0'], counting['1']

    def split(self, attribute_values, attribute_type):
        # attribute generate all posible left, right of a node
        # 0 binary, 1: orinal, 2: nominal
        # results = list of all potential outcomes: (left, right) tuple
        results = []
        if attribute_type == 0:
            #             print(attribute_values)
            if len(attribute_values) == 2:
                g_left = [attribute_values[0]]
                g_right = [attribute_values[1]]
                results.append((g_left, g_right))
        elif attribute_type == 1:
            # co the sort cai list xong roi increse dan dan
            # get all values of this attributes in order
            attribute_values = sorted(attribute_values)
            for i in range(1, len(attribute_values)):
                g_left = attribute_values[:i]
                g_right = attribute_values[i:]
                results.append((g_left, g_right))
        else:
            # gen combinations for left and set all - set(left)
            for L in range(1, len(attribute_values)):
                for com in combinations(attribute_values, L):
                    g_left = list(com)
                    g_right = list(set(attribute_values) - set(com))
                    results.append((g_left, g_right))
        return results

    def stop(self, df, atts):
        # case 1: no more attribute
        count_0, count_1 = self.counting_label(df)
        #         print(count_0, count_1)
        if len(atts) == 0:
            print("Doesn't have any attributes left.")
            return True
        # case 2: node is pure
        elif count_0 == 0 or count_1 == 0:
            print("Pure node.")
            return True
        # case 3: support below min sup
        elif count_0 < self.min_sup or count_1 < self.min_sup:
            #             print(count_0, count_1)
            print(f"Pos: {count_1}, Neg: {count_0} | Support below min sup: {self.min_sup}.")
            return True
        return False

    def find_best_split(self, df, attributes, attribute_type_list):
        # calculate gini of parent node
        parent_gini = self.gini_index(df)
        #         print(parent_gini)
        best_gini_list = []
        best_split_attrs = []
        for attribute, attribute_type in zip(attributes, attribute_type_list):
            #             print(f'attribute: {attribute}, type: {attribute_type}')
            # for each attribute find all potential outcomes
            N = len(df[attribute].index)
            attribute_values = self.dataTrain[attribute].unique().tolist()
            #             print(attribute)
            p_splits = self.split(attribute_values, attribute_type)
            # init value for choosing best split
            max_gain = -100
            best_split_att = None
            gini_list = []

            for p_split in p_splits:
                left, right = p_split
                # calculate gini for this split
                # get sub df for left group
                df_left = df.loc[df[attribute].isin(left)]
                gini_left = self.gini_index(df_left)

                # get sub df for right group
                df_right = df.loc[df[attribute].isin(right)]
                gini_right = self.gini_index(df_right)

                # calculate weighted gini for each split and gain
                weighted_gini = (len(df_left.index) * gini_left + len(df_right.index) * gini_right) / N

                gain_gini = parent_gini - weighted_gini
                gini_list.append(gain_gini)

                if gain_gini > max_gain:
                    max_gain = gain_gini
                    best_split_att = (left, right)
            best_gini_list.append(max_gain)
            best_split_attrs.append(best_split_att)

        best_gini = max(best_gini_list)
        best_attribute = attributes[best_gini_list.index(best_gini)]
        best_split = best_split_attrs[best_gini_list.index(best_gini)]
        return best_split, best_gini, best_attribute

    def classify(self, df):
        # classify by major voting
        count_0, count_1 = self.counting_label(df)
        return '1' if count_1 > count_0 else '0'

    def save_model(self, root, model_path):
        # travesal the tree using bfs
        with open(model_path, 'w') as f:
            queue = []
            queue.append(root)
            ans = []
            while queue:
                len_level = len(queue)
                ans_lv = []
                for i in range(len_level):
                    node = queue.pop(0)
                    ans_lv.append(node.id)
                    if node:
                        line = ''
                        if node.label == -1:
                            parent_id = 'NULL' if node.parent_id is None else str(node.parent_id)
                            if parent_id == 'NULL':
                                line += f'n{node.id}:{parent_id}:{node.splitting_att}'
                            else:
                                line += f'n{node.id}:n{node.parent_id}:{node.splitting_att}'
                            if node.left:
                                left_attr_values = ','.join(node.g_left)
                                line += f':{left_attr_values}:n{node.left.id}'
                                queue.append(node.left)
                            if node.right:
                                right_attr_values = ','.join(node.g_right)
                                line += f':{right_attr_values}:n{node.right.id}'
                                queue.append(node.right)
                            line += f':{node.used_atts}\n'
                        else:
                            class_label = '>50K' if node.label == '1' else '<=50K'
                            line = f'n{node.id}:n{node.parent_id}:leaf:{class_label}\n'
                        #                         print(line)
                        f.write(line)
                print(len(ans_lv))
                print(ans_lv)

    # def load_model(self, model_path):
    #     root = None
    #     return root

    def getClass(self, root, row):
        if root.label == '0' or root.label == '1':
            return root.label
        attr = root.splitting_att
        g_left = root.g_left
        g_right = root.g_right
        if row[attr] in g_left:
            val = self.getClass(root.left, row)

        elif row[attr] in g_right:
            val = self.getClass(root.right, row)
        # else:
        #     print("Attribute: ", attr)
        #     print("Left: ", g_left)
        #     print('right:', g_right)
        #     print("Val: ", row[attr])
        return val

    def predict(self, root, test_data):

        list_data = []
        for index, row in test_data.iterrows():
            data_dict = {}
            data_dict['True'] = row['income']
            data_dict['Prediction'] = '>50K' if self.getClass(root, row) == '1' else '<=50K'
            list_data.append(data_dict)
        fields = ['True', 'Prediction']

        with open('prediction.csv', 'w') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames=fields)

            # writing headers (field names)
            writer.writeheader()

            # writing data rows
            writer.writerows(list_data)

    def tree_grow(self, df, atts, type_atts, parent_id):
        # print(self.numNode, parent_id)
        if self.stop(df, atts):
            leaf = Node()
            leaf.id = self.numNode
            leaf.parent_id = parent_id
            leaf.label = self.classify(df)
            print("Creating a leaf with: ", leaf.label)
            return leaf
        else:
            # calculate parent attribute

            b_split, b_gini, split_attribute = self.find_best_split(df, atts, type_atts)
            g_left, g_right = b_split

            df_left = df.loc[df[split_attribute].isin(g_left)].drop(split_attribute, axis=1)  # data for left child
            df_right = df.loc[df[split_attribute].isin(g_right)].drop(split_attribute,
                                                                      axis=1)  # data for right child# data for right child

            g_left = self.order2category(g_left, split_attribute)
            g_right = self.order2category(g_right, split_attribute)
            print(f"After splitting using {split_attribute}: left: {g_left}, right: {g_right}, gini: {b_gini}")

            remain_atts, remain_types = [], []
            for k, v in zip(atts, type_atts):
                if k != split_attribute:
                    remain_atts.append(k)
                    remain_types.append(v)
            used = ''
            for att in list(self.dataTrain)[:-1]:
                used += '0' if att in remain_atts else '1'

            # count_neg, count_pos = self.counting_label(df)

            # create a new node
            root = Node()
            root.parent_id = parent_id
            root.id = self.numNode
            root.splitting_att = split_attribute
            root.g_left = g_left
            root.g_right = g_right
            # root.n_pos = count_pos
            # root.n_neg = count_neg
            root.used_atts = ''.join(used)
            self.numNode += 1
            # recursive to the left
            print("GO LEFT: ", df_left.shape, self.numNode)
            root.left = self.tree_grow(df_left,
                                       remain_atts,
                                       remain_types,
                                       root.id)
            self.numNode += 1
            # recursive to the right
            print("GO RIGHT: ", df_right.shape, self.numNode)
            root.right = self.tree_grow(df_right,
                                        remain_atts,
                                        remain_types,
                                        root.id)
        return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, help="path to train data")
    parser.add_argument('--model_file', type=str, help="path to model file")
    parser.add_argument('--min_freq', type=int, help="min frequency of node")
    args = parser.parse_args()

    train_data = pd.read_csv(args.train_file)
    # print(train_data.shape)
    # convert label 2 binary
    train_data['income'] = train_data['income'].replace(['<=50K', '>50K'], ['0', '1'])
    # convert ordinal attribute to int
    mapper_age = {'young': "0", 'adult': "1", 'senior': "2", 'old': "3"}
    train_data = mapper(train_data, mapper_age)
    mapper_edu = {'BeforeHS': "0", 'HS-grad': "1" , 'AfterHS': "2", 'Bachelors': "3", 'Grd': "4"}
    train_data = mapper(train_data, mapper_edu)
    mapper_hr = {'part-time': "0", 'full-time': "1", 'over-time': "2"}
    train_data = mapper(train_data, mapper_hr)
    # build tree
    tree = BinaryDT(min_sup=args.min_freq, data_train=train_data)
    attribute_type_list = [1, 2, 1, 2, 2, 2, 0, 0, 0, 1, 0]
    root = tree.tree_grow(train_data, list(train_data), attribute_type_list, None)
    # save model
    tree.save_model(root, args.model_file)