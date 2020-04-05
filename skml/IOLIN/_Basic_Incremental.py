import math
import numpy as np
import pandas as pd
from scipy import stats

from skmultiflow.data import SEAGenerator
from skml import IfnClassifier
from skml.IOLIN import MetaLearning
import skml.Utils as Utils



class BasicIncremental:

    def __init__(self,
                 classifier:IfnClassifier,
                 path,
                 number_of_classes=2,
                 n_min=378,
                 n_max=math.inf,
                 alpha=0.99,
                 Pe=0.5,
                 init_add_count=10,
                 inc_add_count=50,
                 max_add_count=100,
                 red_add_count=75,
                 min_add_count=1,
                 max_window=1000,
                 data_stream_generator=SEAGenerator()):

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

        self.classifier = classifier
        self.path = path
        self.number_of_classes = number_of_classes
        self.n_min = n_min
        self.n_max = n_max
        self.alpha = alpha
        self.Pe = Pe
        self.init_add_count = init_add_count
        self.inc_add_count = inc_add_count
        self.max_add_count = max_add_count
        self.red_add_count = red_add_count
        self.min_add_count = min_add_count
        self.max_window = max_window
        self.window = None
        self.meta_learning = MetaLearning(alpha, number_of_classes)
        self.data_stream_generator = data_stream_generator
        self.data_stream_generator.prepare_for_use()

    @property
    def classifier (self):
        return self._classifier

    @classifier.setter
    def classifier(self, value:IfnClassifier):
        self._classifier = value

    def IN_controll(self):

        self.window = self.meta_learning.calculate_Wint(self.Pe)
        i = self.n_min - self.window
        j = self.window
        add_count = self.init_add_count
        X_batch = []
        y_batch = []

        while i < j:
            X, y = self.data_stream_generator.next_sample()
            X_batch.append(X[0])
            y_batch.append(y[0])
            i = i + 1

        X_batch_df = pd.DataFrame(X_batch)
        self.classifier.fit(X_batch_df, y_batch)
        j = j + self.window
        X_batch.clear()
        y_batch.clear()

        while j < self.n_max:

            while i < j:
                X, y = self.data_stream_generator.next_sample()
                X_batch.append(X[0])
                y_batch.append(y[0])
                i = i + 1

            k = j + add_count
            X_validation_samples = []
            y_validation_samples = []

            while j < k:
                X_validation, y_validation = self.data_stream_generator.next_sample()
                X_validation_samples.append(X_validation[0])
                y_validation_samples.append(y_validation[0])
                j = j + 1

            j = k

            self.incremental_IN(X_batch, y_batch, X_validation_samples, y_validation_samples, add_count)
            i = j - self.window

    def incremental_IN(self,
                       training_window_X,
                       training_window_y,
                       validation_window_X,
                       validation_window_y,
                       add_count):

        Etr = self.classifier.calculate_error_rate(X=training_window_X,
                                                   y=training_window_y)

        Eval = self.classifier.calculate_error_rate(X=validation_window_X,
                                                    y=validation_window_y)

        max_diff = self.meta_learning.get_max_diff(Etr=Etr,
                                                   Eval=Eval,
                                                   add_count=add_count)

        if abs(Eval - Etr) < max_diff: # concept is stable
            self._update_current_network(training_window_X=training_window_X,
                                         training_window_y=training_window_y)
        else: # concept drift detected
            self._induce_new_model(training_window_X=training_window_X,
                                   training_window_y=training_window_y)


    def _update_current_network(self, training_window_X, training_window_y):

        self._check_split_validation(training_window_X=pd.DataFrame(training_window_X),
                                     training_window_y=training_window_y)

        is_last_layer_changed, sec_best_att = self._check_replacement_of_last_layer(training_window_X=training_window_X,
                                                                                    training_window_y=training_window_y)

        if is_last_layer_changed:
            # TODO replace last layer with sec_best_att
            self._new_split_process(training_window_X=training_window_X,
                                    training_window_y=training_window_y)

    def _check_split_validation(self, training_window_X, training_window_y):

        un_significant_nodes = []
        curr_layer = self.classifier.network.root_node.first_layer

        while curr_layer is not None:
            nodes = curr_layer.nodes
            for node in nodes:
                if not node.is_terminal:
                    X, y = Utils.drop_records(X=training_window_X,
                                              y=training_window_y,
                                              attribute_index=curr_layer.index,
                                              value=node.inner_index)

                    attribute_data = list(X[:, curr_layer.index])
                    unique_values = np.unique(attribute_data)

                    conditional_mutual_information = \
                        self.classifier.calculate_conditional_mutual_information(X=training_window_X,
                                                                                 y=training_window_y)

                    statistic = 2 * np.log(2) * len(training_window_y) * conditional_mutual_information
                    critical = stats.chi2.ppf(self.alpha, ((self.number_of_classes - 1) * (len(unique_values) - 1)))

                    if critical < statistic:
                        continue
                    else:
                        un_significant_nodes.append(node)
            self.classifier.set_terminal_nodes(nodes=un_significant_nodes,
                                               class_count=self.classifier.class_count)

    def _check_replacement_of_last_layer(self, training_window_X, training_window_y):

        should_replace = False
        conditional_mutual_information_first_best_att = self.classifier.last_layer_mi
        conditional_mutual_information_second_best_att = 0

        if conditional_mutual_information_first_best_att < conditional_mutual_information_second_best_att:
            should_replace = True

        return should_replace, conditional_mutual_information_second_best_att

    def _new_split_process(self, training_window_X, training_window_y):
        pass

    def _induce_new_model(self, training_window_X, training_window_y):
        self.classifier = self.classifier.fit(training_window_X, training_window_y)
