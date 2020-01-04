import pandas as pd
from sklearn.metrics import accuracy_score

from skmultiflow import IfnClassifier
from skmultiflow import _csvConveter
from skmultiflow._dataProcessing import DataProcessor
import pickle

clf = IfnClassifier(0.99)


def test_old_version(file_path_train, file_path_test):
    data = _csvConveter.CsvConverter.convert(file_path_train)
    # ADD COLS TO FIT FUNCTION IN _ifnClassifier
    clf.fit(data[0], data[1], data[2])
    clf.add_training_set_error_rate(data[0], data[1])
    clf.network.create_network_structure_file()

    z = _csvConveter.CsvConverter.convert_predict(file_path_test)

    y_pred = clf.predict(z)

    # print("accuracy", accuracy_score(z, y_pred))

    #print(clf.predict_proba(z))


#test_old_version("datasets/Credit_full.csv", "datasets/pred_credit.csv")
# test_old_version("datasets/Glass_train.csv", "datasets/Glass_test.csv")


def test_with_dataProcessing(file_path, test_size):
    clf = IfnClassifier(0.9)
    dp = DataProcessor()
    X_train, X_test, y_train, y_test = dp.convert(file_path, test_size)

    clf.fit(X_train, y_train)

    clf.add_training_set_error_rate(X_train, y_train)

    clf.network.create_network_structure_file()

    y_pred = clf.predict(X_test)
    print("accuracy", accuracy_score(y_test, y_pred))

    # print(clf.predict_proba(X_test))


test_with_dataProcessing("datasets/Glass.csv", 0.3)
