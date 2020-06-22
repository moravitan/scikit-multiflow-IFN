import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skml.multi.ifn_classifier_multi import IfnClassifierMulti
from skml.multi.data_processing_multi import DataProcessorMulti
from skmultiflow.data.multilabel_generator import MultilabelGenerator
import filecmp
import numpy as np
import pandas as pd
import shutil

from sklearn import datasets

alpha = 0.95


def test_internet_usage():
    columns_type = ['int64', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                    'category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                    'category', 'category', 'category', 'int64', 'category', 'category', 'category', 'category',
                    'category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                    'category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                    'category', 'category', 'category', 'category', 'category', 'category', 'category', 'category',
                    'category', 'int64']
    clf = IfnClassifierMulti(columns_type, "C:\\Users\\user\PycharmProjects\scikit-multiflow-IFN\skml\\tests\\", alpha)
    dp = DataProcessorMulti()
    X, y = dp.convert("C:\\Users\\user\Desktop\פרויקט גמר\MultiFlow\IFN\Internet_Usage_multi\\final_general2_multi.csv",
                      0.1)
    clf.fit(X, y)
    clf.network.create_network_structure_file(
        path="C:\\Users\\user\PycharmProjects\scikit-multiflow-IFN\skml\\tests\\network.txt")

    assert clf.network.root_node.first_layer.index == 40
    assert clf.network.root_node.first_layer.next_layer.index == 0
    expected_train_error = 0.3116521249648185
    assert np.isclose(expected_train_error, clf.training_error)
    assert len(clf.network.root_node.first_layer.nodes) == 3
    assert len(clf.network.root_node.first_layer.next_layer.nodes) == 161


def test_partial_fit():
    columns_type = ['int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64']

    stream = MultilabelGenerator(n_features=7, n_targets=4, random_state=2)

    estimator = IfnClassifierMulti(columns_type,
                                   "C:\\Users\איתן אביטן\PycharmProjects\scikit-multiflow-IFN\skml\\tests", alpha,
                                   multi_label=False, window_size=100)
    stream.prepare_for_use()
    X, y = stream.next_sample(100)
    estimator.partial_fit(X, y, num_of_classes=stream.n_targets)



    cnt = 0
    max_samples = 500
    predictions = {}
    true_labels = {}
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        # next sample return empty arrays
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            # append to df
            predictions[cnt] = (estimator.predict(X)).to_numpy()
            true_labels[cnt] = y[0]
            correct_predictions_targets = sum(predictions[cnt] == true_labels[cnt])
            for i in range(0, len(correct_predictions_targets)):
                if correct_predictions_targets[i] == 1:
                    correct_predictions += 1
        estimator.partial_fit(X, y, num_of_classes=stream.n_targets)
        cnt += 1

    performance = correct_predictions / (len(predictions.keys()) * stream.n_targets)
    expected_predictions = {100: np.array([[0., 0., 0., 0., ]]), 200: np.array([[1., 0., 0., 0.]]),
                            300: np.array([[0., 1., 0., 0.]]), 400: np.array([[1., 0., 0., 0.]])}

    expected_correct_predictions = 7
    expected_performance = 0.5833333333333334

    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions
    for key in predictions.keys():
        assert np.array_equal(predictions[key], expected_predictions[key])

    print("finish check data stream")


test_partial_fit()
