import math
import copy
import numpy as np
import pandas as pd
from scipy import stats

from skmultiflow.data import SEAGenerator
from skml import IfnClassifier
from skml.IOLIN import MetaLearning
import skml.Utils as Utils
from skml._ifn_network import HiddenLayer


class BasicIncremental:

    def __init__(self,
                 classifier: IfnClassifier,
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
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value: IfnClassifier):
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

        if abs(Eval - Etr) < max_diff:  # concept is stable
            self._update_current_network(training_window_X=training_window_X,
                                         training_window_y=training_window_y)
        else:  # concept drift detected
            self._induce_new_model(training_window_X=training_window_X,
                                   training_window_y=training_window_y)

    def _update_current_network(self, training_window_X, training_window_y):

        copy_network = self._check_split_validation(training_window_X=pd.DataFrame(training_window_X),
                                                    training_window_y=training_window_y)

        should_replace, significant_nodes_indexes = self._check_replacement_of_last_layer(copy_network=copy_network)

        if should_replace:
            self._replace_last_layer(significant_nodes_indexes=significant_nodes_indexes)
            self._new_split_process(training_window_X=training_window_X,
                                    training_window_y=training_window_y)

    def _check_split_validation(self, training_window_X, training_window_y):

        copy_network = self._clone_network(training_window_X=training_window_X,
                                           training_window_y=training_window_y)

        curr_layer = copy_network.root_node.first_layer
        un_significant_nodes = []
        un_significant_nodes_indexes = []

        while curr_layer is not None:
            for node in curr_layer.nodes:
                if node.is_terminal is False:
                    statistic, critical, cmi = \
                        self._calculate_conditional_mutual_information_of_current_window(node=node,
                                                                                         index=curr_layer.index)

                    if critical < statistic: # significant
                        continue
                    else:
                        un_significant_nodes_indexes.append(node.index)
                        un_significant_nodes.append(node)

            self.classifier.set_terminal_nodes(nodes=un_significant_nodes,
                                               class_count=self.classifier.class_count)

            BasicIncremental.eliminate_nodes(nodes=set(un_significant_nodes_indexes),
                                             layer=curr_layer.next_layer,
                                             prev_layer=curr_layer)
            curr_layer = curr_layer.next_layer

        return copy_network

    def _clone_network(self, training_window_X, training_window_y):

        copy_network = copy.copy(self.classifier.network)
        training_window_X_copy = training_window_X.copy()
        training_window_y_copy = training_window_y.copy()

        curr_layer = copy_network.root_node.first_layer
        is_first_layer = True
        nodes_data = {}
        while curr_layer is not None:
            for node in curr_layer.nodes:
                if is_first_layer:
                    if curr_layer.is_continuous:
                        Utils.convert_numeric_values(chosen_split_points=curr_layer.split_points,
                                                     chosen_attribute=curr_layer.index,
                                                     partial_X=training_window_X_copy)

                    partial_X, partial_y = Utils.drop_records(X=training_window_X_copy,
                                                              y=training_window_y_copy,
                                                              attribute_index=curr_layer.index,
                                                              value=node.inner_index)
                    node.partial_X = partial_X
                    node.partial_y = partial_y
                    nodes_data[node.index] = [partial_X, partial_y]

                else:
                    X = nodes_data[node.prev_node][0]
                    y = nodes_data[node.prev_node][1]
                    if curr_layer.is_continuous:
                        Utils.convert_numeric_values(chosen_split_points=curr_layer.split_points,
                                                     chosen_attribute=curr_layer.index,
                                                     partial_X=X)

                    partial_X, partial_y = Utils.drop_records(X=X,
                                                              y=y,
                                                              attribute_index=curr_layer.index,
                                                              value=node.inner_index)
                    node.partial_X = partial_X
                    node.partial_y = partial_y
                    nodes_data[node.index] = [partial_X, partial_y]

        return copy_network

    def _check_replacement_of_last_layer(self, copy_network):

        should_replace = False
        conditional_mutual_information_first_best_att = self.classifier.last_layer_mi
        index_of_second_best_att = self.classifier.index_of_sec_best_att

        if index_of_second_best_att == -1: # There's only one significant attribute
            return should_replace

        conditional_mutual_information_second_best_att, significant_nodes_indexes = \
            self._calculate_cmi_of_sec_best_attribute(copy_network=copy_network,
                                                      sec_best_index=index_of_second_best_att)

        if conditional_mutual_information_first_best_att < conditional_mutual_information_second_best_att:
            should_replace = True

        return should_replace, significant_nodes_indexes

    def _calculate_cmi_of_sec_best_attribute(self, copy_network, sec_best_index):

        conditional_mutual_information = 0
        significant_nodes_indexes = []
        curr_layer = copy_network.root_node.first_layer

        while curr_layer.next.next is not None: # loop until last split
            curr_layer = curr_layer.next_layer

        last_layer_nodes = curr_layer.nodes

        for node in last_layer_nodes:
            statistic, critical, cmi = \
                self._calculate_conditional_mutual_information_of_current_window(node=node,
                                                                                 index=sec_best_index)
            if critical < statistic:  # significant
                conditional_mutual_information += conditional_mutual_information + cmi
                significant_nodes_indexes.append(node.index)

        return conditional_mutual_information, set(significant_nodes_indexes)

    def _calculate_conditional_mutual_information_of_current_window(self, node, index):

        X = node.partial_X
        y = node.partial_y
        attribute_data = list(X[:, index])
        unique_values = np.unique(attribute_data)
        conditional_mutual_information = \
            self.classifier.calculate_conditional_mutual_information(X=attribute_data,
                                                                     y=y)

        statistic = 2 * np.log(2) * len(y) * conditional_mutual_information
        critical = stats.chi2.ppf(self.alpha, ((self.number_of_classes - 1) * (len(unique_values) - 1)))

        return statistic, critical, conditional_mutual_information

    def _replace_last_layer(self, significant_nodes_indexes):

        curr_layer = self.classifier.network.root_node.first_layer
        is_continuous = self.classifier.sec_att_split_points is not None
        index_of_sec_best_att = self.classifier.index_of_sec_best_att

        while curr_layer.next_layer.next_layer is not None:  # loop until last split
            curr_layer = curr_layer.next_layer

        new_layer_nodes = []
        terminal_nodes = []
        last_layer_nodes = curr_layer.nodes
        curr_node_index = max([node.index for node in curr_layer.nodes]) + 1

        for node in last_layer_nodes:
            if node.index in significant_nodes_indexes:
                if is_continuous:
                    Utils.convert_numeric_values(chosen_split_points=self.classifier.sec_att_split_points,
                                                 chosen_attribute=index_of_sec_best_att,
                                                 partial_X=node.partial_X)

                unique_values = np.unique(list(node.partial_X[:, index_of_sec_best_att]))

                for i in unique_values: # create nodes for each unique value
                    attribute_node = Utils.create_attribute_node(partial_X=node.partial_X,
                                                                 partial_y=node.partial_y,
                                                                 chosen_attribute_index=index_of_sec_best_att,
                                                                 attribute_value=i,
                                                                 curr_node_index=curr_node_index,
                                                                 prev_node_index=node.index)
                    new_layer_nodes.append(attribute_node)
                    curr_node_index += curr_node_index + 1

            terminal_nodes.append(node)

        # create and link the new last layer to the network
        new_last_layer = HiddenLayer(index_of_sec_best_att)
        new_last_layer.is_continuous = is_continuous

        if new_last_layer.is_continuous is True:
            new_last_layer.split_points = self.classifier.sec_att_split_points

        new_last_layer.nodes = new_layer_nodes
        curr_layer.next_layer = new_last_layer

        # set all the nodes to be terminals
        self.classifier.set_terminal_nodes(nodes=terminal_nodes,
                                           class_count=self.classifier.class_count)

    def _new_split_process(self, training_window_X, training_window_y):
        pass

    def _induce_new_model(self, training_window_X, training_window_y):
        self.classifier = self.classifier.fit(training_window_X, training_window_y)

    @staticmethod
    def eliminate_nodes(nodes, layer, prev_layer):
        next_layer_nodes = []
        nodes_to_eliminate = []

        if nodes is None or len(nodes) == 0 or layer is None:
            return

        curr_layer_nodes = layer.nodes
        for node in curr_layer_nodes:
            if node.prev_node in nodes:
                nodes_to_eliminate.append(node)
                next_layer_nodes.append(node)

        for node in nodes_to_eliminate:
            layer.nodes.remove(node)

        if len(layer.nodes) == 0:
            prev_layer.next_layer = layer.next_layer

        BasicIncremental.eliminate_nodes(nodes=next_layer_nodes,
                                         layer=layer.next_layer,
                                         prev_layer=layer)
