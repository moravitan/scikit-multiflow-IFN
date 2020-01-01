import pandas as pd

from skmultiflow import IfnClassifier
from skmultiflow import _csvConveter

clf = IfnClassifier(0.999)

#data = _csvConveter.CsvConverter.convert("Chess.csv")

df = pd.read_csv("Chess.csv")
y = df['Class'].values
X = df.drop(['Class'], axis = 1)

X_without_cols_name = []
for index, row in df.iterrows():
    # insert each sample in df to x
    record = [elem for elem in row]
    X_without_cols_name.append(record)

clf.fit(X, y)
clf.add_training_set_error_rate(X_without_cols_name, y)
clf.network.create_network_structure_file()


z = _csvConveter.CsvConverter.convert_predict("Chess_test.csv")
# -------------- predict will return the classes and write it to file --------------

# print(clf.predict(z)

# -------------- predict_proba will return the probability for every class and write it to file --------------
print(clf.predict_proba(z))

