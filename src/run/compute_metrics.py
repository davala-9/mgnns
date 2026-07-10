#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ----
"""
from numpy import arange
from numpy import trapezoid
from numpy import nan_to_num
from src.utils.data_parser import parse
import os.path
from src.utils.utils import check


def precision(tp, fp): # Precision: true positives / positives
    value = 0
    try:
        value = tp / (tp + fp)
    except:
        value = float("NaN")
    finally:
        return value

def recall(tp, fn): # Recall: true positives / positive examples
    value = 0
    try:
        value = tp / (tp + fn)
    except:
        value = float("NaN")
    finally:
        return value

def accuracy(tp, fp, tn, fn): # Accuracy: correct predictions
    value = 0
    try:
        value = (tn + tp) / (tp + fp + tn + fn)
    except:
        value = float("NaN")
    finally:
        return value

def f1score(tp, fp, fn):
    value = 0
    try:
        value = tp / (tp + 0.5 * (fp + fn))
    except:
        value = float("NaN")
    finally:
        return value

def auprc(precision_vector, recall_vector):
    return -1 * trapezoid(precision_vector, recall_vector)


def compute_metrics(predictions, positive_examples, negative_examples, metrics_file):

    threshold_list = [0,0.0000000001, 0.000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001,
                      0.001] + arange(0.01, 1, 0.01).tolist() + [0.999, 0.9999, 0.99999, 0.999999, 0.9999999,
                                                                 0.99999999, 0.999999999, 0.9999999999, 0.99999999999]
    threshold_list = [round(elem, 10) for elem in threshold_list]

    # Each threshold is mapped to a 4-tuple containing true and false positives and negatives.
    threshold_to_counter = {}
    for threshold in threshold_list:
        threshold_to_counter[threshold] = [0, 0, 0, 0]
    entry_for = {"true_positives": 0, "false_positives": 1, "true_negatives": 2, "false_negatives": 3}

    # Facts that do not appear in our predictions file are automatically assigned a score of 0

    # Positive examples
    for (s, p, o) in parse(check(positive_examples, "Positive examples")):
        for threshold in threshold_list:
            if predictions.get((s, p, o),0) > threshold: # True positive
                threshold_to_counter[threshold][entry_for["true_positives"]] += 1
            else: # False negative
                threshold_to_counter[threshold][entry_for["false_negatives"]] += 1

    # Negative examples
    for (s, p, o) in parse(check(negative_examples, "Negative examples")):
        for threshold in threshold_list:
            if predictions.get((s, p, o), 0) > threshold: # False positive
                threshold_to_counter[threshold][entry_for["false_positives"]] += 1
            else: # True negative
                threshold_to_counter[threshold][entry_for["true_negatives"]] += 1

    #  Compute and print result
    recall_vector = []
    precision_vector = []
    with open(metrics_file, 'w') as f:
        f.write("Threshold" + '\t' + "Precision" + '\t' + "Recall" + '\t' + "Accuracy" + '\t' + "F1 Score" + '\n')
        for threshold in threshold_to_counter:
            tp, fp, tn, fn = threshold_to_counter[threshold]
            f.write("{}\t{}\t{}\t{}\t{}\n".format(threshold, precision(tp, fp),
                                                  recall(tp, fn), accuracy(tp, fp, tn, fn),
                                                  f1score(tp, fp, fn)))
            recall_vector.append(recall(tp, fn))
            precision_vector.append(precision(tp, fp))
        # Add extremal points for AUC. This ensures a perfect classifier has AUC 1, a random classifier has AUC 0.5,
        # and an `always wrong' classifier has an AUC 0. Without this, a perfect classifier would have a score of 0!
        precision_vector.insert(0, 0)
        precision_vector.append(1)
        recall_vector.insert(0, 1)
        recall_vector.append(0)
        # Get rid of NaNs
        recall_vector = nan_to_num(recall_vector)
        precision_vector = nan_to_num(precision_vector)
        f.write("Area under precision recall curve: {}\n".format(auprc(precision_vector, recall_vector)))

    f.close()
