import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from skmultiflow import IfnClassifier
from skmultiflow._dataProcessing import DataProcessor


def test_with_dataProcessing(file_path, test_size):
    clf = IfnClassifier(0.9)
    dp = DataProcessor()
    X_train, X_test, y_train, y_test = dp.convert(file_path, test_size)

    clf.fit(X_train, y_train)
    clf.network.create_network_structure_file()

    y_predN = clf.predict(X_test)

    print("accuracy", accuracy_score(y_test, y_predN))
    # print(clf.predict_proba(X_test))


test_with_dataProcessing("datasets/Credit.csv", 0.3)
