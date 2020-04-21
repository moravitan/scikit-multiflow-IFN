import math
import pickle
import os
import numpy as np
import pandas as pd
from skml import IfnClassifier
from skml.IOLIN import MetaLearning, OnlineNetwork
from skmultiflow.data import SEAGenerator


class PureMultiple(OnlineNetwork):

    def __init__(self, classifier: IfnClassifier, path, number_of_classes=2, n_min=378, n_max=math.inf, alpha=0.99,
                 Pe=0.5, init_add_count=10, inc_add_count=50, max_add_count=100, red_add_count=75, min_add_count=1,
                 max_window=1000, data_stream_generator=SEAGenerator(random_state=1)):

        """
        Parameters
        ----------
        classifier :
        path : String
            A path to save the model.
        number_of_classes : int
            The number of classes in the target.
        n_min : int
            The number of the first example to be classified by the system.
        n_max : int
            The number of the last example to be classified by the system.
            (if unspecified, the system will run indefinitely).
        alpha : float
            Significance level
        Pe : float
            Maximum allowable prediction error of the model.
        init_add_count : int
            The number of new examples to be classified by the first model.
        inc_add_count : int
            Amount (percentage) to increase the number of examples between model re-constructions.
        max_add_count : int
            Maximum number of examples between model re-constructions.
        red_add_count : int
            Amount (percentage) to reduce the number of examples between model reconstructions.
        min_add_count : int
            Minimum number of examples between model re-constructions.
        max_window : int
            Maximum number of examples in a training window.
        data_stream_generator : stream generator
            Stream generator for the stream data
        """

        super().__init__(classifier, path, number_of_classes, n_min, n_max, alpha, Pe, init_add_count, inc_add_count,
                         max_add_count, red_add_count, min_add_count, max_window, data_stream_generator)

    def generate(self):
        """ This function is an implementation of Pure Multiple Model IOLIN algorithm as represented
            by Prof. Mark Last, et al. in "https://www.sciencedirect.com/science/article/abs/pii/S156849460800046X".
            This function obtain an IFN model for every window arriving in the stream,
            and validate the prediction on the next window, which represent the validation examples.
            After each iteration the IFN model is saved to a file in the path given by the user.

        """

        counter = 1
        self.window = self.meta_learning.calculate_Wint(self.Pe)
        i = 0
        j = self.window
        add_count = self.init_add_count
        X_batch = []
        y_batch = []

        while j < self.n_max:

            while i < j:
                X, y = self.data_stream_generator.next_sample()
                X_batch.append(X[0])
                y_batch.append(y[0])
                i = i + 1
            X_batch_df = pd.DataFrame(X_batch)

            if os.path.exists(self.path) and len(os.listdir(self.path)) > 0:
                classifier_files_names = os.listdir(self.path)
                generated_classifiers = {}
                for classifier in classifier_files_names:
                    generated_clf = pickle.load(open(self.path + "/" + classifier, "rb"))
                    generated_classifiers[classifier] = abs(generated_clf.calculate_error_rate(X_batch_df, y_batch) -
                                                            classifier.calculate_error_rate(X_batch_df, y_batch))
                chosen_classifier_name = min(generated_classifiers, key=generated_classifiers.get)
                chosen_classifier = pickle.load(open(self.path + "/" + chosen_classifier_name, "rb"))

                Etr = generated_classifiers[chosen_classifier]

                k = j + add_count
                X_validation_samples = []
                y_validation_samples = []

                while j < k:
                    X_validation_sample, y_validation_sample = self.data_stream_generator.next_sample()
                    X_validation_samples.append(X_validation_sample[0])
                    y_validation_samples.append(y_validation_sample[0])
                    j = j + 1

                Eval = self.classifier.calculate_error_rate(X_validation_samples, y_validation_samples)
                max_diff = self.meta_learning.get_max_diff(Etr, Eval, add_count)

                if abs(Eval - Etr) > max_diff:  # concept drift detected
                    chosen_classifier = IfnClassifier(self.alpha)
                    chosen_classifier.fit(X_batch_df, y_batch)
                    path = self.path + "/" + str(counter)
                    pickle.dump(self.classifier, open(path, "wb"))
                    counter = counter + 1

            else:  # cold start
                chosen_classifier = IfnClassifier(self.alpha)
                chosen_classifier.fit(X_batch_df, y_batch)
                path = self.path + "/" + str(counter) + ".pickle"
                pickle.dump(self.classifier, open(path, "wb"))
                counter = counter + 1

            j = j + self.window
            X_batch.clear()
            y_batch.clear()

        last_model = pickle.load(open(self.path + "/" + str(counter - 1) + ".pickle", "rb"))
        return last_model
