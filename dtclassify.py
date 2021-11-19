from dtbuild import Node
import argparse
import pandas as pd
import csv

def travesal(root):
    # using bfs to travesal
    queue = [root]
    ans = []
    while len(queue):
        len_level = len(queue)
        ans_lv = []
        for i in range(len_level):
            node = queue.pop(0)
            if node:
                ans_lv.append(node.id)
                queue.append(node.left)
                queue.append(node.right)
        if len(ans_lv) > 0:
            ans.append(ans_lv)
            print(len(ans_lv), ans_lv)
    return ans


def find_parent(root, parent_id):
    # using bfs to find the parent node of a child
    queue = [root]
    while len(queue):
        len_level = len(queue)
        for i in range(len_level):
            node = queue.pop(0)
            if node:
                if node.id == parent_id:
                    return node
                queue.append(node.left)
                queue.append(node.right)
    return None


def create_node(elements):
    if len(elements) == 4:

        node = Node()
        node.id = int(elements[0][1:])
        node.label = '1' if elements[-1] == '>50K' else '0'
        node.parent_id = int(elements[1][1:])
        return node

    else:

        node_id, parent_id, splitting_att, left_att_val, left_id, right_att_val, right_id = elements
        node = Node()
        node.id = int(node_id[1:])
        node.parent_id = int(parent_id[1:]) if parent_id != 'NULL' else None
        node.splitting_att = splitting_att
        node.g_left = left_att_val.strip().split(',')
        node.left_id = int(left_id[1:])
        node.g_right = right_att_val.strip().split(',')
        node.right_id = int(right_id[1:])
        node.label = -1

        return node


def load_model(model_path):
    root = None
    with open('model.txt', 'r') as f:
        count = 0
        for line in f:
            elements = line.strip().split(':')
            if root is None:
                root = create_node(elements)
            else:
                node = create_node(elements)
                parent = find_parent(root, node.parent_id)

                if parent is not None:
                    if node.id == parent.left_id:
                        parent.left = node
                    elif node.id == parent.right_id:
                        parent.right = node
    return root


def get_class(root, row):
    if root.label == '0' or root.label == '1':
        return root.label

    attr = root.splitting_att
    #     print(attr)
    g_left = root.g_left
    g_right = root.g_right
    #     print(row[attr])
    if row[attr] in g_left:
        val = get_class(root.left, row)

    elif row[attr] in g_right:
        val = get_class(root.right, row)
    else:
        print("Attribute: ", attr)
        print("Left: ", g_left)
        print('right:', g_right)
        print("Val: ", row[attr])
    return val


def predict(root, test_data, save_path):
    #     test_data = pd.read_csv(testdata_path)
    list_data = []
    for index, row in test_data.iterrows():
        data_dict = {}
        data_dict['True'] = row['income']
        data_dict['Prediction'] = '>50K' if get_class(root, row) == '1' else '<=50K'
        list_data.append(data_dict)
    #         print(data_dict)
    #         break
    fields = ['True', 'Prediction']

    with open(save_path, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(list_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help="path to model file")
    parser.add_argument('--test_file', type=str, default="test.csv", help="path of csv test file")
    parser.add_argument('--predictions', type=str, default="predictions.csv", help="Name of csv prediction file")
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_file)
    root = load_model(args.model_file)

    predict(root, test_df, args.predictions)