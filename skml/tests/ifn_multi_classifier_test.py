import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from skml.multi_target.ifnClassifier_multi import IfnClassifier
from skml.multi_target.dataProcessing_multi import DataProcessor


def test_ifnClassifier(file_path, test_size, multi_label=False):
    clf = IfnClassifier(0.9, multi_label)
    dp = DataProcessor()
    X_train, X_test, y_train, y_test = dp.convert(file_path, test_size)

    clf.fit(X_train, y_train)
    # pickle.dump(clf, open("clf.pickle", "wb"))
    # loadedCLF = pickle.load(open("clf.pickle", "rb"))
    # loadedCLF.network.create_network_structure_file()
    # y_predN = loadedCLF.predict(X_test)
    print("finish build multi network")
    y_predN = clf.predict(X_test)
    print("finish predict multi network")
    # did not work with Glass_multi file - multi class multi output
    print("accuracy", accuracy_score(y_test, y_predN))
    print("finish accuracy multi network")
    print(clf.predict_proba(X_test))
    # print("finish predict probability multi network")


test_ifnClassifier("../tests/datasets/Chess_multi.csv", 0.3, multi_label=True)
