"""
Author: Gil Rotem and Tzuriel Hagian
Original code and method by: Prof' Mark Last
License: BSD 3 clause
"""
import numpy as np
from ._ifn_network import IfnNetwork, AttributeNode, HiddenLayer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats
import math
import collections
import time
import sys
import pandas as pd





class IfnClassifierN():
    attributes_array = []
    update_attributes_array = []
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    alpha : float, default='0.99'
        A parameter used for the significance level of the likelihood-ratio tests.
    """

    def __init__(self, alpha=0.99):
        if 0 <= alpha < 1:
            self.alpha = alpha
        else:
            raise ValueError("Enter a valid alpha between 0 to 1")
        self.network = IfnNetwork()

    def _is_numeric(self, X):
        if len(np.unique(X)) == 2:
            return False

    def _get_columns_type(self, X):
        columns_type = []
        for dt in X.columns:
            if len(np.unique(X[dt])) > 10:
                columns_type.append(str(X[dt].dtype))
            else:
                columns_type.append("category")
        return columns_type

    def fit(self, X, y, sample_weight = None):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        start = time.time()
        f = open("outputN.txt", "w+")
        f.write('Output data for dataset: \n\n')

        print('Building the network...')
        global total_records
        global all_nodes_continuous_atts_data
        global attribute_node_mi_data
        global nodes_info_per_attribute
        global split_points
        global split_points_per_node
        global num_of_classes
        global unique_values_per_attribute
        global all_continuous_vars_data
        global is_two_classes

        cols = list(X.columns.values)
        columns_type = self._get_columns_type(X)

        X, y = check_X_y(X, y, accept_sparse=True)

        total_records = len(y)
        unique, counts = np.unique(np.array(y), return_counts=True)
        class_count = np.asarray((unique, counts)).T
        num_of_classes = len(unique)
        is_two_classes = num_of_classes == 2

        nodes_info_per_attribute = {}
        # map that contains all the split points for all attributes
        split_points = {}
        split_points_per_node = {}
        # create list of all attributes
        attributes_indexes = list(range(0, len(X[0])))

        f.write('Total instances: ' + str(total_records) + '\n')
        f.write('Number of candidate input attributes is: ' + str(len(attributes_indexes)) + '\n')
        f.write('Minimum confidence level is: ' + str(self.alpha) + '\n\n')
        f.close()

        self.network.build_target_layer(unique)

        attributes_mi = {}
        unique_values_per_attribute = {}
        is_first_layer = True
        curr_node_index = 1
        current_layer = None
        partial_X = X
        partial_y = y
        significant_attributes_per_node = {}
        layer = 'first'

        while len(attributes_indexes) > 0:
            chosen_split_points = []
            if current_layer is not None:
                i = 0
                significant_attributes_per_node = {}
                max_cmi = 0
                global_chosen_attribute = -1
                for node in current_layer.get_nodes():
                    partial_X = node.partial_x
                    partial_y = node.partial_y
                    chosen_attribute_index, attributes_mi = self._choose_split_attribute(partial_X,
                                                                                         partial_y,
                                                                                         attributes_indexes,
                                                                                         columns_type)
                    significant_attributes_per_node[i] = attributes_mi
                    i = i + 1
                    if max_cmi < attributes_mi[chosen_attribute_index]:
                        max_cmi = attributes_mi[chosen_attribute_index]
                        global_chosen_attribute = chosen_attribute_index
                        if 'category' not in columns_type[global_chosen_attribute]:
                            chosen_split_points = split_points[global_chosen_attribute]
            # first layer
            else:
                global_chosen_attribute, attributes_mi = self._choose_split_attribute(partial_X,
                                                                                      partial_y,
                                                                                      attributes_indexes,
                                                                                      columns_type)
            # will need it for future calculations
            current_unique_values_per_attribute = unique_values_per_attribute.copy()

            # there isn't an attribute to split by
            if global_chosen_attribute == -1:
                if curr_node_index == 1:
                    print('No Nodes at the network. choose smaller alpha')
                    sys.exit()
                if not is_first_layer:
                    attributes_mi = self._calculate_total_mi(significant_attributes_per_node)
                self._write_details_to_file(layer,
                                            attributes_mi,
                                            global_chosen_attribute,
                                            cols[global_chosen_attribute])
                break

            # create new hidden layer of the maximal mutual information attribute and set the layer nodes
            nodes_list = []

            is_continuous = 'category' not in columns_type[global_chosen_attribute]
            unique_values_per_node = {}
            # if chosen att is continuous we convert the partial x values by the splits values
            if is_continuous:
                self._convert_numeric_values(chosen_split_points,
                                             global_chosen_attribute,
                                             current_layer,
                                             partial_X)

            un_significant_nodes = []
            prev_node = 0
            if not is_first_layer:
                prev_node = current_layer.index

            for i in unique_values_per_attribute[global_chosen_attribute]:
                if current_layer is not None:
                    node_index = 0
                    for node in current_layer.get_nodes():
                        attributes_mi_per_node = significant_attributes_per_node[node_index]
                        # if global chosen attribute is significant at this node
                        if attributes_mi_per_node[global_chosen_attribute] > 0:
                            partial_X = node.partial_x
                            partial_y = node.partial_y
                            attribute_node = self._create_attribute_node(partial_X,
                                                                         partial_y,
                                                                         global_chosen_attribute,
                                                                         i,
                                                                         curr_node_index,
                                                                         prev_node)
                            nodes_list.append(attribute_node)
                            curr_node_index += 1
                        else:
                            un_significant_nodes.append(node)
                        node_index = node_index + 1
                # first layer
                else:
                    attribute_node = self._create_attribute_node(partial_X,
                                                                 partial_y,
                                                                 global_chosen_attribute,
                                                                 i,
                                                                 curr_node_index,
                                                                 prev_node)
                    nodes_list.append(attribute_node)
                    curr_node_index += 1

            next_layer = HiddenLayer(global_chosen_attribute)

            if is_first_layer:
                self.network.root_node.first_layer = next_layer
            else:
                current_layer.next_layer = next_layer

            next_layer.set_nodes(nodes_list)
            if is_continuous:
                next_layer.is_continuous = True
                next_layer.split_points = chosen_split_points

            un_significant_nodes_set = list(set(un_significant_nodes))
            for node in un_significant_nodes_set:
                node.set_terminal()
                # add weight to terminal node
                node.set_weight_probability_pair(calc_weight(node.partial_y, class_count, total_records))

            current_layer = next_layer

            if not is_first_layer:
                attributes_mi = self._calculate_total_mi(significant_attributes_per_node)

            self._write_details_to_file(layer,
                                        attributes_mi,
                                        global_chosen_attribute,
                                        cols[global_chosen_attribute])
            layer = 'next'
            is_first_layer = False

            attributes_indexes.remove(global_chosen_attribute)
            split_points = {key: [] for key in split_points}
            nodes_info_per_attribute = {}
            split_points_per_node = {}

        for node in current_layer.get_nodes():
            node.set_terminal()
            # add weight to terminal node
            node.set_weight_probability_pair(calc_weight(node.partial_y, class_count, total_records))

        with open('outputN.txt', 'a') as f:
            f.write('Total nodes created:' + str(curr_node_index) + "\n")
            end = time.time()
            f.write("Running time: " + str(round(end - start, 3)) + " Sec")
            f.close()

        # `fit` should always return `self`
        self.is_fitted_ = True
        print("Done. Network is fitted")
        return self

    def _choose_split_attribute(self, X, y, attributes_indexes, columns_type):
        attributes_mi = {}
        # get the attribute that holds the maximal mutual information
        for attribute_index in attributes_indexes:
            attribute_data = list(X[:, attribute_index])
            unique_values_per_attribute[attribute_index] = np.unique(attribute_data)

            # check if attribute is continuous
            is_continuous = 'category' not in columns_type[attribute_index]

            # if feature is of type continuous
            if is_continuous:
                split_points[attribute_index] = []
                data_class_array = list(zip(attribute_data, y))
                data_class_array.sort(key=lambda tup: tup[0])

                nodes_info_per_attribute[attribute_index] = []
                nodes_info_per_attribute[attribute_index].append((0, 0))
                self._discretization(data_class_array, attribute_index)

                if bool(split_points[attribute_index]):  # there are split points
                    sum_of_splits = sum([pair[1] for pair in nodes_info_per_attribute[attribute_index]])
                    attributes_mi[attribute_index] = sum_of_splits
                else:
                    attributes_mi[attribute_index] = 0

            # categorical feature
            else:
                mutual_info_score = calc_MI(attribute_data, y, total_records)
                statistic = 2 * np.log(2) * total_records * mutual_info_score
                critical = stats.chi2.ppf(self.alpha, ((num_of_classes - 1) *
                                                       ((len(unique_values_per_attribute[attribute_index])) - 1)
                                                       ))
                if critical < statistic:
                    attributes_mi[attribute_index] = mutual_info_score
                else:
                    attributes_mi[attribute_index] = 0

        chosen_attribute_index = max(attributes_mi, key=attributes_mi.get)

        return chosen_attribute_index, attributes_mi

    def _convert_numeric_values(self, chosen_split_points, chosen_attribute, layer, partial_X):

        # for the case it's the first layer
        if not bool(chosen_split_points):
            chosen_split_points = split_points[chosen_attribute]

        unique_values_per_attribute[chosen_attribute] = np.arange(len(chosen_split_points) + 1)
        chosen_split_points.sort()

        if layer is not None:
            for node in layer.get_nodes():
                partial_X = node.partial_x
                for record in partial_X:
                    record[chosen_attribute] = self._find_split_position(record[chosen_attribute],
                                                                         chosen_split_points)
        # first layer
        else:
            for record in partial_X:
                record[chosen_attribute] = self._find_split_position(record[chosen_attribute],
                                                                     chosen_split_points)

    def _calculate_total_mi(self, significant_attributes_per_node):
        attributes_mi = {}
        for index, dict in significant_attributes_per_node.items():
            for key, value in dict.items():
                if key in attributes_mi.keys():
                    attributes_mi[key] += value
                else:
                    attributes_mi[key] = value
        return attributes_mi

    def _create_attribute_node(self, partial_X, partial_y, chosen_attribute, i, curr_node_index, prev_node):
        x_y_tuple = drop_records(partial_X,
                                 chosen_attribute,
                                 partial_y,
                                 i)
        # add the relevant records to each node
        attributes_node = AttributeNode(curr_node_index,
                                        i,
                                        prev_node,
                                        chosen_attribute,
                                        x_y_tuple[0],
                                        x_y_tuple[1])
        return attributes_node

    def _write_details_to_file(self, layer_number, attributes_mi, chosen_attribute_index, chosen_attribute):
        with open('outputN.txt', 'a') as f:
            f.write(layer_number + ' layer attribute: \n')
            for index, mi in attributes_mi.items():
                f.write(str(index) + ': ' + str(round(mi, 3)) + '\n')

            if chosen_attribute_index != -1:
                f.write('\nChosen attribute is: ' + chosen_attribute + "(" + str(chosen_attribute_index) + ")" + '\n\n')
                # f.write('\nSplit points are: ' + str(split_points[chosen_attribute]) + '\n\n')
            else:
                f.write('\nChosen attribute is: None' + "(" + str(chosen_attribute_index) + ")" + '\n\n')
            f.close()

    def _find_split_position(self, record, positions):
        # smaller than the first
        if record < positions[0]:
            return 0
        # equal-larger than the last
        if record >= positions[len(positions) - 1]:
            return len(positions)

        for i in range(len(positions)):
            first_position = positions[i]
            second_position = positions[i + 1]
            if first_position <= record < second_position:
                return i + 1




    def _discretization(self, interval, attribute_index):

        interval_values = [i[0] for i in interval]
        distinct_attribute_data = np.unique(interval_values)

        split_point_mi_map = {}

        iterator = iter(distinct_attribute_data)
        next(iterator)
        for T in iterator:
            t_attribute_data = []
            new_y = []
            for data_class_tuple in interval:
                new_y.append(data_class_tuple[1])
                if data_class_tuple[0] < T:
                    t_attribute_data.append(0)
                else:
                    t_attribute_data.append(1)

            if is_two_classes:
                critical = stats.chi2.ppf(self.alpha, (num_of_classes - 1))
            else:
                rel_num_of_classes = len(np.unique(np.array(new_y)))
                critical = stats.chi2.ppf(self.alpha, (rel_num_of_classes - 1))
            t_mi = calc_MI(t_attribute_data, new_y, total_records)
            statistic = 2 * np.log(2) * total_records * t_mi

            # t in attribute is a possible split point
            if critical < statistic:
                # for each point save it's mutual information
                split_point_mi_map[T] = t_mi

        if bool(split_point_mi_map):  # if not empty
            split_point = max(split_point_mi_map, key=split_point_mi_map.get)
            if split_point not in split_points[attribute_index]:
                split_points[attribute_index].append(split_point)
            if attribute_index not in nodes_info_per_attribute:
                nodes_info_per_attribute[attribute_index] = []

            for node_tuple in nodes_info_per_attribute[attribute_index]:
                if node_tuple[0] == 0:
                    new_node_tuple = (0, node_tuple[1] + split_point_mi_map[split_point])
                    nodes_info_per_attribute[attribute_index].append(new_node_tuple)
                    nodes_info_per_attribute[attribute_index].remove(node_tuple)
                    break

            sub_interval_smaller_equal = []
            sub_interval_larger = []
            for elem in interval:
                if elem[0] < split_point:
                    sub_interval_smaller_equal.append(elem)
                else:
                    sub_interval_larger.append(elem)

            if bool(sub_interval_smaller_equal):
                self._discretization(sub_interval_smaller_equal, attribute_index)
            if bool(sub_interval_larger):
                self._discretization(sub_interval_larger, attribute_index)

    def _discretization_second(self, interval, attribute_index, distinct_attribute_data, nodes):

        split_point_mi_map = {}
        node_T_significants = {}
        for node in nodes:
            node_T_significants[node.index] = {}

        iterator = iter(distinct_attribute_data)
        next(iterator)

        for T in iterator:
            for node in nodes:
                t_attribute_data = []
                new_y = []
                # X = list(np.array(np.array(node.partial_x))[:, attribute_index])
                # data_class_array = list(zip(X, node.partial_y))
                for x, y in interval[node.inner_index]:
                    new_y.append(y)
                    if x < T:
                        t_attribute_data.append(0)
                    else:
                        t_attribute_data.append(1)

                if is_two_classes:
                    critical = stats.chi2.ppf(self.alpha, (num_of_classes - 1))
                else:
                    rel_num_of_classes = len(np.unique(np.array(new_y)))
                    critical = stats.chi2.ppf(self.alpha, (rel_num_of_classes - 1))
                t_mi = calc_MI(t_attribute_data, new_y, total_records)
                statistic = 2 * np.log(2) * total_records * t_mi

                # t in attribute is a possible split point
                if critical < statistic:
                    # for each point save it's mutual information
                    if T in split_point_mi_map.keys():
                        split_point_mi_map[T] += t_mi
                    else:
                        split_point_mi_map[T] = t_mi

                    split_points_per_node[node.index] = []
                    split_points_per_node[node.index].append(T)
                    node_T_significants[node.index][T] = t_mi

        if bool(split_point_mi_map):  # if not empty
            split_point = max(split_point_mi_map, key=split_point_mi_map.get)
            if split_point not in split_points[attribute_index]:
                split_points[attribute_index].append(split_point)
            if attribute_index not in nodes_info_per_attribute:
                nodes_info_per_attribute[attribute_index] = []

            new_tuple_list = []
            for node_tuple in nodes_info_per_attribute[attribute_index]:
                if split_point in node_T_significants[node_tuple[0]].keys():
                    new_node_tuple = (node_tuple[0], node_tuple[1] + node_T_significants[node_tuple[0]][split_point])
                    new_tuple_list.append(new_node_tuple)
                else:
                    new_tuple_list.append(node_tuple)

            nodes_info_per_attribute[attribute_index] = new_tuple_list

            sub_interval_smaller_equal = {}
            sub_interval_larger = {}
            i = 0
            for list in interval:
                sub_interval_smaller_equal[i] = []
                sub_interval_larger[i] = []
                for elem in list:
                    if elem[0] < split_point:
                        sub_interval_smaller_equal[i].append(elem)
                    else:
                        sub_interval_larger[i].append(elem)
                i = i + 1

            if len(sub_interval_smaller_equal) > 0:
                self._discretization_second(sub_interval_smaller_equal, attribute_index, nodes)
            if len(sub_interval_larger) > 0:
                self._discretization_second(sub_interval_larger, attribute_index, nodes)

    def _recursive_split_points(self, interval, attribute_index, distinct_attribute_data, nodes):

        sub_data_map = {}
        node_T_significants = {}
        for node in nodes:
            node_T_significants[node.index] = {}

        iter_att = iter(distinct_attribute_data)
        next(iter_att)

        if is_two_classes:
            critical = stats.chi2.ppf(self.alpha, (num_of_classes - 1))

        for T in iter_att:
            for node in nodes:
                t_attribute_data = []
                new_y = []
                for data_class_tuple in interval[node.inner_index]:
                    if data_class_tuple[0] < T:
                        t_attribute_data.append(0)
                    else:
                        t_attribute_data.append(1)

                    new_y.append(data_class_tuple[1])

                if not is_two_classes:
                    rel_num_of_classes = len(np.unique(np.array(new_y)))
                    critical = stats.chi2.ppf(self.alpha, (rel_num_of_classes - 1))

                t_mi = calc_MI(t_attribute_data, new_y, total_records)
                statistic = 2 * np.log(2) * total_records * t_mi

                if critical < statistic:
                    if T in sub_data_map.keys():
                        sub_data_map[T] += t_mi
                    else:
                        sub_data_map[T] = t_mi

                    node_T_significants[node.index][T] = t_mi

        if bool(sub_data_map):
            split_point = max(sub_data_map, key=sub_data_map.get)
            if split_point not in split_points[attribute_index]:
                split_points[attribute_index].append(split_point)
            if attribute_index not in nodes_info_per_attribute:
                nodes_info_per_attribute[attribute_index] = []

            new_tuple_list = []
            for node_tuple in nodes_info_per_attribute[attribute_index]:
                if split_point in node_T_significants[node_tuple[0]].keys():
                    new_node_tuple = (node_tuple[0], node_tuple[1] + node_T_significants[node_tuple[0]][split_point])
                    new_tuple_list.append(new_node_tuple)
                else:
                    new_tuple_list.append(node_tuple)

            nodes_info_per_attribute[attribute_index] = new_tuple_list

            sub_interval_0 = []
            sub_interval_1 = []

            for list in interval:
                list_temp0 = []
                list_temp1 = []
                for data, target in list:
                    if data < split_point:
                        list_temp0.append((data, target))
                    else:
                        list_temp1.append((data, target))
                sub_interval_0.append(list_temp0)
                sub_interval_1.append(list_temp1)

        if len(sub_interval_0) > 0:
            self._recursive_split_points(sub_interval_0, attribute_index, nodes, distinct_attribute_data)
        if len(sub_interval_1) > 0:
            self._recursive_split_points(sub_interval_1, attribute_index, nodes, distinct_attribute_data)



    def predict(self, X):
        """ A reference implementation   of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        predicted = []
        for record in X:
            curr_layer = self.network.root_node.first_layer
            prev_node_index = 0
            found_terminal_node = False
            while curr_layer is not None and not found_terminal_node:
                record_value = record[curr_layer.index]
                if curr_layer.is_continuous is not False:
                    record_value = self._find_split_position(record_value, curr_layer.split_points)
                for node in curr_layer.nodes:
                    if node.inner_index == record_value and node.prev_node == prev_node_index:
                        chosen_node = node
                        if chosen_node.is_terminal:
                            max_weight = -math.inf
                            predicted_class = -math.inf
                            for class_index, weight_prob_pair in chosen_node.weight_probability_pair.items():
                                if weight_prob_pair[0] > max_weight:
                                    max_weight = weight_prob_pair[0]
                                    predicted_class = class_index
                            predicted.append(predicted_class)
                            found_terminal_node = True
                        else:
                            curr_layer = curr_layer.next_layer
                            prev_node_index = chosen_node.index
                        break

        index = 1
        with open('predictN.txt', 'w') as f:
            for row in predicted:
                f.write(str(index) + '. ' + str(row) + '\n')
                index += 1
            f.close()

        return np.array(predicted)

