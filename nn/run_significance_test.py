#!/usr/bin/env python

"""
Usage:

python run_significance_test.py predictions1 predictions2 gold_answers

"""
import sys
import numpy as np
from statsmodels.sandbox.stats.runs import mcnemar as mcnemar_test
import csv

np.random.seed(123)

def compute_mcnemar_test(contingency_table):
    assert np.shape(contingency_table) == (2, 2)
    statistic, p_value = mcnemar_test(contingency_table, exact=True, correction=True)
    return statistic, p_value

def read_submission_file(fn):
    """
    Reads a file with labels in the submission format: story_id, label
    :param fn: filename with predictions
    :return: numpy array of shape = (1, 2) with the number of positive/negative labels
    """
    labels = []

    with open(fn) as csv_file:
        reader = iter(csv.reader(csv_file, delimiter=",", quotechar='"'))
        header = next(reader)

        if not header[0].isalpha(): # if there are no headers in the submission file
            lab = int(header[-1]) - 1
            assert lab in [0,1]
            labels.append(lab)

        for row in reader:
            lab = int(row[-1]) -1
            labels.append(lab)

    return labels

def build_contingency_table(labels1, labels2, gold_arr):
    """
    Our computation of the contingency table.
    :param labels1: an array of label predictions of clf1
    :param labels2: an array of label predictions of clf2
    :param gold_arr: an array of gold labels
    :return: numpy array (contingency table) of shape (2,2):

                    Clf2-label-0    Clf2-label-1

    Clf1-label-0         a                b

    Clf1-label-1         c                d

    """

    pos_pos = pos_neg = neg_pos = neg_neg = 0

    for idx, lab1 in enumerate(labels1):
        lab2 = labels2[idx]
        gold_lab = gold_arr[idx]

        if lab1 == gold_lab:
            if lab2 == gold_lab:
                pos_pos +=1
            else:
                pos_neg +=1

        elif lab2 != gold_lab:
            neg_neg +=1
        else:
            neg_pos +=1

    return np.asarray([[pos_pos, pos_neg],[neg_pos, neg_neg]])

if __name__ == "__main__":
    label_array1 = read_submission_file(sys.argv[1])
    label_array2 = read_submission_file(sys.argv[2])
    gold_array = read_submission_file(sys.argv[3])

    contingency_table = build_contingency_table(label_array1, label_array2, gold_array)
    statistic, p_value = compute_mcnemar_test(contingency_table)

    print "Contingency table: \n", contingency_table
    print "McNemar test statistic: %0.7f\nP-value: %0.7f" % (statistic, p_value)
