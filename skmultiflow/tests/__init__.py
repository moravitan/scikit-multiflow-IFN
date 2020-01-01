import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from skmultiflow import IfnClassifier
from skmultiflow._dataProcessing import DataProcessor

clf = IfnClassifier(0.95)

file_path = "dataset.csv"

# df = pd.read_csv(file_path)
# y = df['Class'].values
# X = df.drop(['Class'], axis = 1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

dp = DataProcessor()

X_train, X_test, y_train, y_test = dp.convert(file_path, 0.3)

clf.fit(X_train, y_train)

X_without_cols_name = []
for index, row in X_train.iterrows():
    # insert each sample in df to x
    record = [elem for elem in row]
    X_without_cols_name.append(record)

clf.add_training_set_error_rate(X_without_cols_name, y_train)

clf.network.create_network_structure_file()




#z = _csvConveter.CsvConverter.convert_predict("Chess_test.csv")
# -------------- predict will return the classes and write it to file --------------

y_pred = clf.predict(X_test)

#print(y_pred)

print(accuracy_score(y_test, y_pred))

# -------------- predict_proba will return the probability for every class and write it to file --------------
print(clf.predict_proba(X_test))

