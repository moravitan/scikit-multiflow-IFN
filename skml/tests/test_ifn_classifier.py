import pickle
from sklearn.metrics import accuracy_score
from skmultiflow.data import RandomTreeGenerator

from skml import IfnClassifier
from skml._data_processing import DataProcessor
import pytest
import os
import filecmp
import numpy as np
import shutil

dataset_path = "C:\\Users\איתן אביטן\PycharmProjects\scikit-multiflow-IFN\skml\\tests\datasets\Credit.csv"
test_size_percentage = 0.3
alpha = 0.99
test_tmp_folder = "tmp"


def _clean_test_env():
    shutil.rmtree(test_tmp_folder, ignore_errors=True)


def _setup_test_env():
    if not os.path.isdir(test_tmp_folder):
        os.mkdir(test_tmp_folder)


# def test_classifier_const_dataset():
#     _setup_test_env()
#     clf = IfnClassifier(alpha)
#     dp = DataProcessor()
#     x_train, x_test, y_train, y_test = dp.convert(dataset_path, test_size_percentage)
#
#     clf.fit(x_train, y_train)
#     clf.network.create_network_structure_file()
#     y_pred = clf.predict(x_test)
#
#     expected_pred = np.array([1, 2, 3, 4, 5])  # maybe change to get from file
#
#     assert filecmp.cmp('tmp/network_structure.txt', 'expert_network_structure.txt') is True
#     assert np.array_equal(y_pred, expected_pred)
#     assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, expected_pred)
#
#     _clean_test_env()


def test__model_pickle_const_dataset():
    _setup_test_env()
    clf = IfnClassifier(alpha)
    dp = DataProcessor()
    x_train, x_test, y_train, y_test = dp.convert(dataset_path, test_size_percentage)

    clf.fit(x_train, y_train)
    pickle.dump(clf, open("tmp/clf.pickle", "wb"))
    clf.network.create_network_structure_file()
    os.rename("tmp/network_structure.txt", "tmp/clf_network_structure.txt")
    y_pred = clf.predict(x_test)

    loaded_clf = pickle.load(open("tmp/clf.pickle", "rb"))
    loaded_clf.network.create_network_structure_file()
    os.rename("tmp/network_structure.txt", "tmp/loaded_clf_network_structure.txt")
    loaded_y_pred = loaded_clf.predict(x_test)

    assert filecmp.cmp('tmp/loaded_clf_network_structure.txt', 'tmp/clf_network_structure.txt') is True
    assert np.array_equal(y_pred, loaded_y_pred)
    print(accuracy_score(y_test, y_pred))
    assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, loaded_y_pred)

    _clean_test_env()


def test_partial_fit():
    stream = RandomTreeGenerator(tree_random_state=1, sample_random_state=1)
    clf = IfnClassifier(alpha=0.99, window_size=100)
    for i in range(0, 10):
        X, y = stream.next_sample(10)
        clf.partial_fit(X, y)


# test__model_pickle_const_dataset()
test_partial_fit()
