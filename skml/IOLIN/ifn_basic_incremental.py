import math
import pickle
import pandas as pd
import os
from skmultiflow.data import SEAGenerator
from skml.IOLIN import IncrementalOnlineNetwork


class BasicIncremental(IncrementalOnlineNetwork):

    def __init__(self, classifier, path, number_of_classes=2, n_min=378, n_max=math.inf, alpha=0.99,
                 Pe=0.5, init_add_count=10, inc_add_count=50, max_add_count=100, red_add_count=75, min_add_count=1,
                 max_window=1000, data_stream_generator=SEAGenerator()):

        super().__init__(classifier, path, number_of_classes, n_min, n_max, alpha, Pe, init_add_count, inc_add_count,
                         max_add_count, red_add_count, min_add_count, max_window, data_stream_generator)

    def generate(self):

        self.window = self.meta_learning.calculate_Wint(self.Pe)
        self.classifier.window_size = self.window
        print("Window size is: " + str(self.window))
        j = self.window
        add_count = self.init_add_count

        while j < self.n_max:

            print("Buffering training sample window from SEA Generator")

            X_batch, y_batch = self.data_stream_generator.next_sample(self.window)

            if self.classifier.is_fitted:  # the network already fitted at least one time before

                print("Buffering validation sample window from SEA Generator")

                X_validation_samples, y_validation_samples = self.data_stream_generator.next_sample(add_count)
                j = j + add_count

                self._incremental_IN(X_batch, y_batch, X_validation_samples, y_validation_samples, add_count)

            else:  # cold start
                print("***Cold Start***")
                self._induce_new_model(training_window_X=X_batch, training_window_y=y_batch)
                j = j + self.window
                print("### Model " + str(self.counter) + " saved ###")

            j = j + self.window

        full_path = os.path.join(self.path, str(self.counter - 1))
        last_model = pickle.load(open(full_path + ".pickle", "rb"))
        print("Model path is: " + str(self.path) + "/" + str(self.counter - 1))
        return last_model

    def _incremental_IN(self,
                        training_window_X,
                        training_window_y,
                        validation_window_X,
                        validation_window_y,
                        add_count):

        Etr = self.classifier.calculate_error_rate(X=training_window_X,
                                                   y=training_window_y)

        print("Current model training error rate for the current window of samples: " + str(Etr))
        Eval = self.classifier.calculate_error_rate(X=validation_window_X,
                                                    y=validation_window_y)
        print("Current model validation error rate for the current window of samples: " + str(Eval))
        max_diff = self.meta_learning.get_max_diff(Etr=Etr,
                                                   Eval=Eval,
                                                   add_count=add_count)

        if abs(Eval - Etr) < max_diff:  # concept is stable
            print("###Concept is stable###")
            print("###Updating the current model###")
            self._update_current_network(training_window_X=training_window_X,
                                         training_window_y=training_window_y)
        else:  # concept drift detected
            print("***Concept drift detected***")
            print("***Generating new model***")
            self._induce_new_model(training_window_X=training_window_X,
                                   training_window_y=training_window_y)
