import pandas as pd
from sklearn.model_selection import train_test_split

"""
NOTES:
    - split test and train randomly
    - add method train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
    - see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    - convert nominal features to dummy/binary
"""


class CsvConverter:
    @staticmethod
    def convert(csv_file_path):
        df = pd.read_csv(csv_file_path)
        # target
        y = df['Class'].values
        x = []
        df = df.drop("Class", axis=1)
        # features
        cols = list(df.columns.values)
        for index, row in df.iterrows():
            # insert each sample in df to x
            record = [elem for elem in row]
            x.append([elem for elem in row])

        return x, y, cols

    @staticmethod
    def convert_predict(csv_file_path):
        df = pd.read_csv(csv_file_path)
        x = []
        for index, row in df.iterrows():
            record = [elem for elem in row]
            x.append(record)

        return x
