import csv
import os
import numpy as np

def load_data(folder):
    with open(os.path.join(folder, 'train.csv'),
            'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        train = [(row[0], int(row[1])) for row in reader]

    with open(os.path.join(folder, 'dev.csv'),
            'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        dev = [(row[0], int(row[1])) for row in reader]
    return (train, dev)


def evaluate(labels, predictions):

    tp = np.sum(np.logical_and(predictions == 1, labels == 1))
    tn = np.sum(np.logical_and(predictions == 0, labels == 0))
    fp = np.sum(np.logical_and(predictions == 1, labels == 0))
    fn = np.sum(np.logical_and(predictions == 0, labels == 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    
    return ((tp, tn, fp, fn), precision, recall, f_score)