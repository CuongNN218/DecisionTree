import numpy as np
import argparse
import seaborn as sns; sns.set_theme()
sns.set(font_scale=2)


def plot_matrix(cm, classes, title):
    ax = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
    ax.set(title=title, xlabel="predicted label", ylabel="true label")


def counter(predictions):
    N = 0
    pos, neg, true, false = 0, 0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    with open(predictions) as f:
        for line in f:
            pred, gt = line.strip().split()
            if pred == 1:
                pos += 1
            else:
                neg += 1
            if gt == 1:
                true += 1
            else:
                false += 1
            if int(pred) == 1 and int(gt) == 1:
                tp += 1
                pos += 1
                true += 1
            elif int(pred) == 1 and int(gt) == 0:
                fp += 1
                pos += 1
                false += 0
            elif int(pred) == 0 and int(gt) == 0:
                fn += 1
                neg += 1
                false += 1
            else:
                tn += 1
                true += 1
                neg += 1
            N += 1

    return pos, neg, true, false, N, tp, tn, fp, fn


def cal_metric(tp, tn, fp, fn):
    err = 0
    acc = 0
    recall = 0
    precision = 0
    f1 = 0
    return err, acc, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, help="Name to prediction file")
    args = parser.parse_args()
    pos, neg, true, false, N, tp, tn, fp, fn = counter(args.pred)
    err, acc, recall, f1 = cal_metric(tp, tn, fp, fn)
    print("Error Rate: ", err)
    print("Accuracty: ", acc)
    print("Recall: ", recall)
    print("F1-score: ", f1)
    val = np.array([[tp, tn], [fp,fn]])
    classes = ['class A', 'class B']
    title = "title example"
    plot_matrix(val, classes, title)