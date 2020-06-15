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

# # data = pd.read_csv(r"skml/tests/datasets/music.csv")
# data = "C:\\Users\איתן אביטן\PycharmProjects\scikit-multiflow-IFN\skml\\tests\datasets\Credit_multi.csv"
# # y = data.iloc[:, :6]
# # X = data.iloc[:, 6:]
# test_size_percentage = 0.3
# alpha = 0.99
# dir = "tmp"
#
#
# def _clean_test_env():
#     shutil.rmtree(dir, ignore_errors=True)
#
#
# def _setup_test_env():
#     os.mkdir(dir)
#
#
# def test__model_pickle_const_dataset_multi_target():
#     # dir = tmpdir.mkdir("multi_target")
#     _setup_test_env()
#     clf = IfnClassifierMulti(alpha, multi_label=False, file_path=dir)
#     dp = DataProcessorMulti()
#     # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage)
#     x_train, x_test, y_train, y_test = dp.convert(data, test_size=test_size_percentage)
#
#     clf.fit(x_train, y_train)
#     # pickle_file = tmpdir.join("clf.pickle")
#     pickle_file = dir.join("clf.pickle")
#     pickle.dump(clf, open(pickle_file, "wb"))
#     # network_structure_file = tmpdir.join("network_structure.txt")
#     network_structure_file = dir.join("network_structure.txt")
#     clf.network.create_network_structure_file(path=network_structure_file)
#     y_pred = clf.predict(x_test)
#
#     loaded_clf = pickle.load(open(pickle_file, "rb"))
#     # pickle_network_structure_file = tmpdir.join("loaded_network_structure.txt")
#     pickle_network_structure_file = dir.join("loaded_network_structure.txt")
#     loaded_clf.network.create_network_structure_file(path=pickle_network_structure_file)
#     loaded_y_pred = loaded_clf.predict(x_test)
#
#     assert filecmp.cmp(pickle_network_structure_file, network_structure_file) is True
#     assert np.array_equal(y_pred, loaded_y_pred)
#     print("accuracy:", accuracy_score(y_test, y_pred))
#     assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, loaded_y_pred)
#     _clean_test_env()


# def test__model_pickle_const_dataset_multi_label(tmpdir):
#     dir = tmpdir.mkdir("multi_label")
#     clf = IfnClassifierMulti(alpha, multi_label=True, file_path=dir)
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage)
#
#     clf.fit(x_train, y_train)
#     pickle_file = tmpdir.join("clf.pickle")
#     pickle.dump(clf, open(pickle_file, "wb"))
#     network_structure_file = tmpdir.join("network_structure.txt")
#     clf.network.create_network_structure_file(path=network_structure_file)
#     y_pred = clf.predict(x_test)
#
#     loaded_clf = pickle.load(open(pickle_file, "rb"))
#     pickle_network_structure_file = tmpdir.join("loaded_network_structure.txt")
#     loaded_clf.network.create_network_structure_file(path=pickle_network_structure_file)
#     loaded_y_pred = loaded_clf.predict(x_test)
#
#     assert filecmp.cmp(pickle_network_structure_file, network_structure_file) is True
#     assert np.array_equal(y_pred, loaded_y_pred)
#     print("accuracy:", accuracy_score(y_test, y_pred))
#     assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, loaded_y_pred)

alpha = 0.95


def test_internet_usage():
    clf = IfnClassifierMulti(alpha)
    dp = DataProcessorMulti()
    X, y = dp.convert("C:\\Users\\user\Desktop\פרויקט גמר\MultiFlow\IFN\Internet_Usage_multi\\final_general2.csv",0.1)
    clf.fit(X, y)
    clf.network.create_network_structure_file(path="C:\\Users\\user\PycharmProjects\scikit-multiflow-IFN\skml\\tests\\network.txt")


# test_internet_usage()


def test_partial_fit():
    stream = MultilabelGenerator(n_samples=100, n_features=20, n_targets=4, n_labels=4, random_state=0)

    estimator = IfnClassifierMulti(alpha, multi_label=True, window_size=100)
    stream.prepare_for_use()
    X, y = stream.next_sample(100)
    estimator.partial_fit(X, y, classes=stream.n_targets)

    cnt = 0
    max_samples = 2000
    predictions = []
    true_labels = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(estimator.predict(X)[0])
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1]):
                correct_predictions += 1

        estimator.partial_fit(X, y)
        cnt += 1

    performance = correct_predictions / len(predictions)
    expected_predictions = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

    expected_correct_predictions = 14
    expected_performance = 0.7368421052631579

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions


# test_partial_fit()

def test_chess():
    clf = IfnClassifierMulti(alpha)
    dp = DataProcessorMulti()
    x_train, x_test, y_train, y_test = dp.convert("C:\\Users\\user\PycharmProjects\scikit-multiflow-IFN\skml\\tests\datasets\Chess_multi.csv",0.3)
    clf.fit(x_train, y_train)
    clf.predict(x_test)


test_chess()
