import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from skml.multi.ifn_classifier_multi import IfnClassifierMulti
from skml.multi._data_processing_multi import DataProcessor
import pytest
import os
import filecmp
import numpy as np
import shutil

dataset_path_1 = "C:\\Users\איתן אביטן\PycharmProjects\scikit-multiflow-IFN\skml\\tests\datasets\Chess_multi.csv"
dataset_path_2 = "C:\\Users\איתן אביטן\PycharmProjects\scikit-multiflow-IFN\skml\\tests\datasets\Credit_multi.csv"
test_size_percentage = 0.3
alpha = 0.99
test_tmp_folder = "tmp_multi"


def _clean_test_env():
    shutil.rmtree(test_tmp_folder, ignore_errors=True)


def _setup_test_env():
    if not os.path.isdir(test_tmp_folder):
        os.mkdir(test_tmp_folder)


def test_classifier_const_dataset(multi_label=False):
    _setup_test_env()
    clf = IfnClassifierMulti(alpha, multi_label)
    dp = DataProcessor()
    x_train, x_test, y_train, y_test = dp.convert(dataset_path_2, test_size_percentage)

    clf.fit(x_train, y_train)
    clf.network.create_network_structure_file()
    y_pred = clf.predict(x_test)

    assert isinstance(y_pred, pd.DataFrame)
    assert accuracy_score(y_test, y_pred) > 0.2

    _clean_test_env()


def test__model_pickle_const_dataset(path, multi_label=False):
    _setup_test_env()
    clf = IfnClassifierMulti(alpha, multi_label)
    dp = DataProcessor()
    x_train, x_test, y_train, y_test = dp.convert(path, test_size_percentage)

    clf.fit(x_train, y_train)
    pickle.dump(clf, open("tmp_multi/clf.pickle", "wb"))
    clf.network.create_network_structure_file()
    os.rename("tmp_multi/network_structure.txt", "tmp_multi/clf_network_structure.txt")
    y_pred = clf.predict(x_test)

    loaded_clf = pickle.load(open("tmp_multi/clf.pickle", "rb"))
    loaded_clf.network.create_network_structure_file()
    os.rename("tmp_multi/network_structure.txt", "tmp_multi/loaded_clf_network_structure.txt")
    loaded_y_pred = loaded_clf.predict(x_test)

    assert filecmp.cmp('tmp_multi/loaded_clf_network_structure.txt', 'tmp_multi/clf_network_structure.txt') is True
    assert np.array_equal(y_pred, loaded_y_pred)
    print("accuracy:", accuracy_score(y_test, y_pred))
    assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, loaded_y_pred)

    _clean_test_env()


test__model_pickle_const_dataset(dataset_path_1, True)
test__model_pickle_const_dataset(dataset_path_2)
