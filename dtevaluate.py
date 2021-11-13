import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

sns.set(font_scale=2)


def plot_matrix(cm, classes, title):
    fig = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False, fmt='g')
    fig.set(title=title, xlabel="predicted label", ylabel="true label")
    fig = fig.get_figure()
    fig.savefig('confusion_matrix.png')
    plt.show()



def counting_label(df, name):
    counting = df[name].value_counts().to_dict()
    return counting['1'], counting['0']


def counter(test_data):
    p_positive, p_negative = counting_label(test_data, 'Prediction')

    n_pos = test_data.loc[test_data['True'] == test_data['Prediction']].shape[0]
    n_sample = test_data.shape[0]
    n_neg = n_sample - n_pos

    true_pos = test_data.loc[(test_data['True'] == '1') & (test_data['Prediction'] == '1')].shape[0]
    true_neg = test_data.loc[(test_data['True'] == '0') & (test_data['Prediction'] == '0')].shape[0]
    false_pos = test_data.loc[(test_data['True'] == '0') & (test_data['Prediction'] == '1')].shape[0]
    false_neg = n_sample - (true_pos + true_neg + false_pos)
    return p_positive, p_negative, n_pos, n_neg, n_sample, true_pos, true_neg, false_pos, false_neg


def cal_metric(tp, tn, fp, fn):
    epsilon = 1e-7  # avoid divide by 0
    error_rate = (fn + fp) / (tp + tn + fp + fn + epsilon)
    accuracy = (tn + tp) / (tp + tn + fp + fn + epsilon)
    recall = tp / (tp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    f1_score = (2 * precision * recall) / (precision + recall + epsilon)
    return error_rate, accuracy, recall, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help="Name to prediction file")
    args = parser.parse_args()

    pred_data = pd.read_csv(args.pred)
    pred_data['True'] = pred_data['True'].replace(['<=50K', '>50K'], ['0', '1'])
    pred_data['Prediction'] = pred_data['Prediction'].replace(['<=50K', '>50K'], ['0', '1'])

    pos, neg, true, false, N, tp, tn, fp, fn = counter(pred_data)

    err, acc, recall, f1 = cal_metric(tp, tn, fp, fn)
    print("Error Rate: ", err)
    print("Accuracty: ", acc)
    print("Recall: ", recall)
    print("F1-score: ", f1)
    val = np.array([[tp, fn], [fp, tn]])
    classes = ['>50K', '<=50K']
    title = "Confusion Matrix"
    plot_matrix(val, classes, title)