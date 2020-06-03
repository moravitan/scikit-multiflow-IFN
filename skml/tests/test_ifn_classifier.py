import pickle
from sklearn.metrics import accuracy_score
from skmultiflow.data import RandomTreeGenerator

from skml import IfnClassifier
from skml.data_processing import DataProcessor
import os
import filecmp
import numpy as np
import shutil

# dataset_path = "C:\\Users\איתן אביטן\PycharmProjects\scikit-multiflow\src\skmultiflow\data\datasets\elec.csv"
dataset_path = r"C:\\Users\איתן אביטן\PycharmProjects\scikit-multiflow-IFN\skml\tests\datasets\Credit.csv"
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
    stream = RandomTreeGenerator(tree_random_state=112, sample_random_state=112)

    estimator = IfnClassifier(alpha)

    X, y = stream.next_sample(150)
    estimator.partial_fit(X, y)

    cnt = 0
    max_samples = 3000
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
    expected_predictions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
                            1.0, 1.0, 1.0]

    expected_correct_predictions = 20
    expected_performance = 0.6896551724137931

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions


test__model_pickle_const_dataset()
# test_partial_fit()
