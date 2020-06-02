import math
import pickle

import pandas as pd
from skmultiflow.data import SEAGenerator

from skml import IncrementalOnlineNetwork, IfnClassifier, utils
from skml._ifn_network import IfnNetwork


class Advanced(IncrementalOnlineNetwork):

    def __init__(self, classifier: IfnClassifier, path, number_of_classes=2, n_min=378, n_max=math.inf, alpha=0.99,
                 Pe=0.5, init_add_count=10, inc_add_count=50, max_add_count=100, red_add_count=75, min_add_count=1,
                 max_window=1000, data_stream_generator=SEAGenerator()):
        super().__init__(classifier, path, number_of_classes, n_min, n_max, alpha, Pe, init_add_count, inc_add_count,
                         max_add_count, red_add_count, min_add_count, max_window, data_stream_generator)

    def generate(self):

        self.window = self.meta_learning.calculate_Wint(self.Pe)
        i = self.n_min - self.window
        j = self.window
        X_batch = []
        y_batch = []

        while j < self.n_max:

            while i < j:
                X, y = self.data_stream_generator.next_sample()
                X_batch.append(X[0])
                y_batch.append(y[0])
                i = i + 1

            if self.classifier.is_fitted:  # the network already fitted at least one time before
                cloned_network = IncrementalOnlineNetwork.clone_network(network=self.classifier.network,
                                                                        training_window_X=X_batch,
                                                                        training_window_y=y_batch)

                self.__network_update(cloned_network=cloned_network)

            else:  # cold start
                X_batch_df = pd.DataFrame(X_batch)
                self.classifier.fit(X_batch_df, y_batch)
                j = j + self.window
                # save the model
                path = self.path + "/" + str(self.counter)
                pickle.dump(self.classifier, open(path, "wb"))
                self.counter = self.counter + 1

            j = j + self.window
            X_batch.clear()
            y_batch.clear()

        last_model = pickle.load(open(self.path + "/" + str(self.counter - 1) + ".pickle", "rb"))
        return last_model

    def __network_update(self, cloned_network: IfnNetwork, training_window_X):

        self.__calculate_mutual_information_of_cloned_network(cloned_network=cloned_network)

        original_curr_layer = self.classifier.network.root_node.first_layer
        cloned_curr_layer = cloned_network.root_node.first_layer

        while original_curr_layer is not None:
            original_network_mutual_information = original_curr_layer.mutual_information
            cloned_network_mutual_information = cloned_curr_layer.mutual_information

            if original_network_mutual_information * 0.95 <= cloned_network_mutual_information:
                continue
            else:
                pass

            if original_curr_layer.next_layer is None:
                self._new_split_process(training_window_X=training_window_X)

    def __calculate_mutual_information_of_cloned_network(self, cloned_network: IfnNetwork):
        curr_layer = cloned_network.root_node.first_layer

        while curr_layer is not None:
            layer_nodes = curr_layer.get_nodes()
            total_mi = 0
            for node in layer_nodes:
                node_mi = self.classifier._calculate_conditional_mutual_information(X=node.partial_x,
                                                                                    y=node.partial_y)
                total_mi += node_mi

            curr_layer.mutual_information = total_mi
