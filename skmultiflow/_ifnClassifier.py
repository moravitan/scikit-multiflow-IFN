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


def calc_MI(x, y, total_records):
    """

    :param x:
    :param y:
    :param total_records:
    :return:
    """
    partial_records = len(y)
    # count the number of classes (0 and 1)
    unique, counts = np.unique(np.array(y), return_counts=True)
    # <class, number_of_appearances>
    class_count = np.asarray((unique, counts)).T
    # count the number of distinct values in x
    unique, counts = np.unique(np.array(x), return_counts=True)
    # <value, number_of_appearances>
    data_count = np.asarray((unique, counts)).T
    data_dic = collections.defaultdict(int)

    # count the number of appearances for each tuple x[i],y[i]
    for i in range(len(y)):
        data_class_tuple = x[i], y[i]
        data_dic[data_class_tuple] = data_dic[data_class_tuple] + 1

    total_mi = 0

    # for each data-class
    # key = [feature_value,class]
    for key, value in data_dic.items():
        # for each class
        curr_class_count = None
        for c_count in class_count:
            if c_count[0] == key[1]:
                curr_class_count = c_count[1]

        # for each data
        curr_data_count = None
        for d_count in data_count:
            if d_count[0] == key[0]:
                curr_data_count = d_count[1]

        joint = value / total_records
        cond = value / partial_records
        cond_x = curr_data_count / partial_records
        cond_y = curr_class_count / partial_records

        mutual_information = joint * math.log(cond / (cond_x * cond_y), 2)

        total_mi += mutual_information
    return total_mi


def calc_weight(y, class_count, total_records):
    """

    :param y: class
    :param class_count: tuple <class, number_of_appearances>
    :param total_records: number of records
    :return:
    """
    weight_per_class = {}
    for class_info in class_count:
        # partial_y = np.extract(y, np.where(y == [class_info[0]]), axis=0)
        cut_len = len(np.extract(y == [class_info[0]], y))
        if cut_len != 0:
            weight = (cut_len / total_records) * (math.log((cut_len / len(y)) / (class_info[1] / total_records), 2))
            weight_per_class[class_info[0]] = (weight, (cut_len / len(y)))
        else:
            weight_per_class[class_info[0]] = (0, 0)
    return weight_per_class


def drop_records(X, atr_index, y, node_index):
    """

    :param X: records from data frame
    :param atr_index: feature index
    :param y: class
    :param node_index: unique value in X[][atr_index]
    :return:
    """
    new_x = []
    new_y = []
    for i in range(len(y)):
        if X[i][atr_index] == node_index:
            new_x.append(X[i])
            new_y.append(y[i])
    return np.array(new_x), np.array(new_y)


def _convert_X(X):
    """

    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
    :return: {list} of X
    """
    x = []
    for index, row in X.iterrows():
        # insert each sample in df to x
        x.append([elem for elem in row])
    return x


class IfnClassifier():
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

    def fit(self, X, y):
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
        cols = list(X.columns.values)
        f = open("output.txt", "w+")
        f.write('Output data for dataset: \n\n')
        columns_type = []
        # for dt in X.dtypes:
        #    columns_type.append(str(dt))

        for dt in X.columns:
            if len(np.unique(X[dt])) > 10:
                columns_type.append(str(X[dt].dtype))
            else:
                columns_type.append("category")

        # check_X_y - scikit-learn function.
        # includes multi_output param (can be assign to True while implementing multi label
        X, y = check_X_y(X, y, accept_sparse=True)
        print('Building the network...')
        global total_records
        global all_nodes_continuous_atts_data
        global attribute_node_mi_data
        global nodes_info_per_attribute
        global split_points
        global num_of_classes
        global unique_values_per_attribute
        global all_continuous_vars_data
        global is_two_classes
        total_records = len(y)
        unique, counts = np.unique(np.array(y), return_counts=True)
        class_count = np.asarray((unique, counts)).T

        nodes_info_per_attribute = {}
        # map that contains all the split points for all attributes
        split_points = {}

        # continuous and categorical attributes
        continuous_attributes_type = {}

        # create list of all attributes
        updated_attributes_array = list(range(0, len(X[0])))

        f.write('Total instances: ' + str(total_records) + '\n')
        f.write('Number of candidate input attributes is: ' + str(len(updated_attributes_array)) + '\n')
        f.write('Minimum confidence level is: ' +str( self.alpha) + '\n\n')
        num_of_classes = len(np.unique(y))

        if num_of_classes == 2:
            is_two_classes = True
        else:
            is_two_classes = False

        self.network.build_target_layer(np.unique(y))

        attributes_mi = {}
        unique_values_per_attribute = {}
        curr_node_index = 1

        # get the attribute that holds the maximal mutual information
        for attribute in updated_attributes_array:
            attribute_data = []
            """ MOR
            Maybe can be simplified by using pandas.
            for each attribute extract the entire col
            """
            #for record in X: #REFACTORED
            #    attribute_data.append(record[attribute])
            """ OMRI
            Done what MOR wrote
            """
            attribute_data = list(X[:, attribute])

            unique_values_per_attribute[attribute] = np.unique(attribute_data)

            # check if attribute is continuous

            if len(np.unique(attribute_data)) <= 10 or 'category' in columns_type[attribute]:
                continuous_attributes_type[attribute] = False
            else:
                continuous_attributes_type[attribute] = True
                split_points[attribute] = []

            # if feature is of type continuous
            if continuous_attributes_type[attribute]:
                data_class_array = list(zip(attribute_data, y))
                # data_class_array = []
                # for i in range(len(attribute_data)):
                #    data_class_array.append((attribute_data[i], y[i]))
                data_class_array.sort(key=lambda tup: tup[0])

                nodes_info_per_attribute[attribute] = []
                nodes_info_per_attribute[attribute].append((0, 0))
                first_loop_recursive_split_points(data_class_array, attribute, self)

                if not bool(split_points[attribute]):  # there are not split points
                    attributes_mi[attribute] = 0
                else:
                    sum_of_splits = sum([pair[1] for pair in nodes_info_per_attribute[attribute]])
                    attributes_mi[attribute] = sum_of_splits
            # categorial feature
            else:
                mutual_info_score = calc_MI(attribute_data, y, total_records)
                statistic = 2 * np.log(2) * total_records * mutual_info_score
                critical = stats.chi2.ppf(self.alpha, ((num_of_classes - 1) *
                                                       ((len(unique_values_per_attribute[attribute])) - 1)
                                                       ))
                if critical < statistic:
                    attributes_mi[attribute] = mutual_info_score
                else:
                    attributes_mi[attribute] = 0

        chosen_attribute = max(attributes_mi, key=attributes_mi.get)
        if attributes_mi[chosen_attribute] == 0:
            print('No Nodes at the network. choose smaller alpha')
            sys.exit()
        updated_attributes_array.remove(chosen_attribute)


        f.write('first layer attribute: \n')
        for index, mi in attributes_mi.items():
            f.write(str(index) + ': ' + str(round(mi, 3)) + '\n')

        if chosen_attribute != -1:
            f.write('\nChosen attribute is: ' + cols[chosen_attribute] + "(" + str(chosen_attribute) + ")" + '\n\n')
            # f.write('\nSplit points are: ' + str(split_points[chosen_attribute]) + '\n\n')
        else:
            f.write('\nChosen attribute is: None' + "(" + str(chosen_attribute) + ")" + '\n\n')

        # will need it for future calculations
        current_unique_values_per_attribute = unique_values_per_attribute.copy()

        # create new hidden layer of the maximal mutual information attribute and set the layer nodes
        first_layer = HiddenLayer(chosen_attribute)
        self.network.root_node.first_layer = first_layer
        nodes_list = []

        # if chosen att is continuous we convert the partial x values their positions by the splits values
        if continuous_attributes_type[chosen_attribute]:
            unique_values_per_attribute[chosen_attribute] = np.arange(len(split_points[chosen_attribute]) + 1)

            split_points[chosen_attribute].sort()

            for record in X:
                returned_split_index = find_split_position(record[chosen_attribute],
                                                           split_points[chosen_attribute])
                record[chosen_attribute] = returned_split_index

        for i in unique_values_per_attribute[chosen_attribute]:
            x_y_tuple = drop_records(X, chosen_attribute, y, i)
            # add the relevant records to each node
            nodes_list.append(AttributeNode(curr_node_index, i, 0, chosen_attribute, x_y_tuple[0], x_y_tuple[1]))
            curr_node_index += 1
        first_layer.set_nodes(nodes_list)
        current_layer = first_layer
        if continuous_attributes_type[chosen_attribute]:
            current_layer.is_continuous = split_points[chosen_attribute]

        # initialize map values to empty lists
        split_points = {key: [] for key in split_points}

        part_continuous_vars_data = {}
        while len(updated_attributes_array) > 0:
            # get the attribute that holds the maximal mutual information
            nodes_info_per_attribute = {}
            all_nodes_continuous_atts_data = {}
            attribute_node_mi_data = {}
            all_continuous_vars_data = {}

            # for each attribute
            for attribute in updated_attributes_array:
                part_continuous_vars_data[attribute] = {}
                if attribute not in nodes_info_per_attribute:
                    nodes_info_per_attribute[attribute] = []

                # continuous attribute
                if continuous_attributes_type[attribute]:
                    if attribute in all_nodes_continuous_atts_data:
                        all_nodes_att_map = all_nodes_continuous_atts_data[attribute]
                    else:
                        all_nodes_att_map = {}
                    if attribute not in all_continuous_vars_data:
                        node_continuous_vars_data = {}
                        all_continuous_vars_data[attribute] = node_continuous_vars_data

                    distinct_attribute_data = current_unique_values_per_attribute[attribute]
                    iter_att = iter(distinct_attribute_data)
                    next(iter_att)
                    part_continuous_vars_data[attribute][0] = {}
                    part_continuous_vars_data[attribute][1] = {}
                    for node in current_layer.nodes:
                        data_class_array = []

                        part_continuous_vars_data[attribute][0][node.index] = []

                        for i in range(node.partial_y.size):
                            data_class_array.append((node.partial_x[i][attribute], node.partial_y[i]))

                        data_class_array.sort(key=lambda tup: tup[0])
                        all_continuous_vars_data[attribute][node.index] = data_class_array

                    if is_two_classes:
                        critical = stats.chi2.ppf(self.alpha, (num_of_classes - 1))

                    for T in iter_att:
                        # for each node in current layer
                        for node in current_layer.nodes:
                            t_attribute_data = []
                            new_y = []
                            for data_class_tuple in all_continuous_vars_data[attribute][node.index]:
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
                                if attribute not in attribute_node_mi_data.keys():
                                    attribute_node_mi_data[attribute] = {}

                                if node.index not in attribute_node_mi_data[attribute].keys():
                                    attribute_node_mi_data[attribute][node.index] = {}

                                attribute_node_mi_data[attribute][node.index][T] = t_mi
                                if T in all_nodes_att_map:
                                    all_nodes_att_map[T] += t_mi
                                else:
                                    all_nodes_att_map[T] = t_mi

                            all_nodes_continuous_atts_data[attribute] = all_nodes_att_map

                    if bool(all_nodes_continuous_atts_data[attribute]):
                        split_point = max(all_nodes_continuous_atts_data[attribute],
                                          key=all_nodes_continuous_atts_data[attribute].get)
                        split_points[attribute].append(split_point)
                        for node in current_layer.nodes:
                            part_continuous_vars_data[attribute][0][node.index] = []
                            part_continuous_vars_data[attribute][1][node.index] = []

                            # If the node z is split by the threshold T max, mark the node as split
                            if node.index in attribute_node_mi_data[attribute].keys():
                                if split_point in attribute_node_mi_data[attribute][node.index].keys():
                                    node_info_tuple = (node.index, attribute_node_mi_data[attribute][node.index][split_point])
                                    nodes_info_per_attribute[attribute].append(node_info_tuple)

                                else:
                                    new_node_tuple = (node.index, 0)
                                    nodes_info_per_attribute[attribute].append(new_node_tuple)
                            else:
                                new_node_tuple = (node.index, 0)
                                nodes_info_per_attribute[attribute].append(new_node_tuple)

                            for elem in all_continuous_vars_data[attribute][node.index]:
                                if elem[0] < split_point:
                                    part_continuous_vars_data[attribute][0][node.index].append(elem)
                                else:
                                    part_continuous_vars_data[attribute][1][node.index].append(elem)

                        recursive_split_points(part_continuous_vars_data[attribute][0], attribute, self,
                                                 current_layer.nodes, distinct_attribute_data)

                        recursive_split_points(part_continuous_vars_data[attribute][1], attribute, self,
                                                 current_layer.nodes, distinct_attribute_data)
                else:
                    # for each node in current layer
                    for node in current_layer.nodes:
                        attribute_data = []
                        """
                        Simplify using pandas
                        """
                        for record in node.partial_x:
                            attribute_data.append(record[attribute])
                        node_mi = calc_MI(attribute_data, node.partial_y, total_records)
                        statistic = 2 * np.log(2) * total_records * node_mi
                        critical = stats.chi2.ppf(self.alpha, ((num_of_classes - 1) *
                                                               ((len(unique_values_per_attribute[attribute])) - 1)
                                                               ))
                        if critical < statistic:
                            node_info_tuple = (node.index, node_mi)
                        else:
                            node_info_tuple = (node.index, 0)

                        nodes_info_per_attribute[attribute].append(node_info_tuple)

            max_node_mi = 0
            chosen_index = -1

            # extract attribute with maximum MI
            for attribute_index in nodes_info_per_attribute:
                node_mi = 0
                for node_info in nodes_info_per_attribute[attribute_index]:
                        node_mi += node_info[1]
                if node_mi > max_node_mi:
                    max_node_mi = node_mi
                    chosen_index = attribute_index

            chosen_attribute = chosen_index

            f.write('next layer attribute: \n')
            for index, mi_list in nodes_info_per_attribute.items():
                curr_mi = sum(mi[1] for mi in mi_list)
                f.write(str(index) + ': ' + str(round(curr_mi, 3)) + '\n')

            if chosen_attribute != -1:
                f.write('\nChosen attribute is: ' + cols[chosen_attribute] + "(" + str(chosen_attribute) + ")" + '\n\n')
                # f.write('\nSplit points are: ' + str(split_points[chosen_attribute]) + '\n\n')


            else:
                f.write('\nChosen attribute is: None' + "(" + str(chosen_attribute) + ")" + '\n\n')

            # stop building the network if all layer's nodes are terminal
            if chosen_attribute == -1:
                break

            # set terminal nodes
            for node_tuple in nodes_info_per_attribute[chosen_attribute]:
                if node_tuple[1] == 0:  # means chi2 test didnt pass
                    node = current_layer.get_node(node_tuple[0])
                    if node is not None:
                        node.set_terminal()
                        # add weight to terminal node
                        node.set_weight_probability_pair(calc_weight(node.partial_y, class_count, total_records))

            nodes_info_per_attribute = {}  # initialize max nodes mi data
            nodes_list = []
            for curr_layer_node in current_layer.nodes:
                if not curr_layer_node.is_terminal:

                    # if chosen att is continuous we convert the partial x values their positions by the splits values
                    if continuous_attributes_type[chosen_attribute]:
                        unique_values_per_attribute[chosen_attribute] = np.\
                            arange(len(split_points[chosen_attribute]) + 1)

                        split_points[chosen_attribute].sort()

                        for record in curr_layer_node.partial_x:
                            returned_split_index = find_split_position(record[chosen_attribute],
                                                                       split_points[chosen_attribute])
                            record[chosen_attribute] = returned_split_index

                    for i in unique_values_per_attribute[chosen_attribute]:
                        x_y_tuple = drop_records(curr_layer_node.partial_x,
                                                 chosen_attribute, curr_layer_node.partial_y, i)
                        # create a node only if it has data
                        if len(x_y_tuple[0]):
                            nodes_list.append(AttributeNode(curr_node_index, i, curr_layer_node.index,
                                                            chosen_attribute, x_y_tuple[0], x_y_tuple[1]))
                            curr_node_index += 1

            updated_attributes_array.remove(chosen_attribute)

            # update unique values for each attribute:
            current_unique_values_per_att = {}
            for node in current_layer.nodes:
                if not node.is_terminal:
                    for att in updated_attributes_array:
                        if continuous_attributes_type[att]:
                            data_array = []
                            for node_data in all_continuous_vars_data[att].values():
                                for data_tuple in node_data:
                                    data_array.append(data_tuple[0])

                            current_unique_values_per_att[att] = np.unique(np.array(data_array))

            current_unique_values_per_attribute = current_unique_values_per_att
            new_layer = HiddenLayer(chosen_attribute)
            current_layer.next_layer = new_layer
            current_layer = new_layer
            current_layer.set_nodes(nodes_list)
            if continuous_attributes_type[chosen_attribute]:
                current_layer.is_continuous = split_points[chosen_attribute]

            # initialize map values to empty lists
            split_points = {key: [] for key in split_points}

        # that means we used all of the attributes so we have to set the last layer's nodes to be terminal
        # or all nodes in current layer are terminal
        if len(updated_attributes_array) == 0 or chosen_attribute == -1:
            for node in current_layer.nodes:
                node.set_terminal()
                # add weight to terminal node
                node.set_weight_probability_pair(calc_weight(node.partial_y, class_count, total_records))

        f.write('Total nodes created:' + str(curr_node_index) + "\n")
        end = time.time()
        f.write("Running time: " + str(round(end - start, 3)) + " Sec")
        f.close()

        # `fit` should always return `self`
        self.is_fitted_ = True
        print("Done. Network is fitted")
        return self

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
                    record_value = find_split_position(record_value, curr_layer.is_continuous)
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
        with open('predict.txt', 'w') as f:
            for row in predicted:
                f.write(str(index) + '. ' + str(row) + '\n')
                index += 1
            f.close()

        return np.array(predicted)

    def predict_proba(self, X):
        """ A reference implementation of a predicting function.

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
                    record_value = find_split_position(record_value, curr_layer.is_continuous)
                for node in curr_layer.nodes:
                    if node.inner_index == record_value and node.prev_node == prev_node_index:
                        chosen_node = node
                        if chosen_node.is_terminal:
                            found_terminal_node = True
                            weights_of_node = []
                            for class_index, weight_prob_pair in chosen_node.weight_probability_pair.items():
                                weights_of_node.append((round(weight_prob_pair[1], 3)))
                            predicted.append(weights_of_node)
                        else:
                            curr_layer = curr_layer.next_layer
                            prev_node_index = chosen_node.index
                        break

        index = 1
        with open('predict.txt', 'w') as f:
            for row in predicted:
                f.write(str(index) + '. ' + str(row) + '\n')
                index += 1
            f.close()

        return np.array(predicted)

    def add_training_set_error_rate(self, x, y):
        correct = 0
        for i in range(len(y)):
            # predicted_value = self.predict([x[i]])[0]
            predicted_value = self.predict(x.iloc[[i]])[0]
            if predicted_value == y[i]:
                correct += 1

        error_rate = (len(y) - correct) / len(y)
        with open('output.txt', 'a') as f:
            f.write("\nError rate is: " + str(round(error_rate, 3)))
            f.close()



def first_loop_recursive_split_points(sub_interval, attribute, self):
    """
    function for continuous feature only
    :param sub_interval: data frame of <value,class>
    :param attribute: column index of a feature
    :param self:
    :return:
    """
    sub_interval_values = [i[0] for i in sub_interval]
    distinct_attribute_data = np.unique(sub_interval_values)

    sub_data_map = {}

    iter_att = iter(distinct_attribute_data)
    next(iter_att)

    if is_two_classes:
        critical = stats.chi2.ppf(self.alpha, (num_of_classes - 1))

    for T in iter_att:
        t_attribute_data = []
        new_y = []
        for data_class_tuple in sub_interval:
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

        # t in attribute is a possible split point
        if critical < statistic:
            # for each point save it's mutual information
            sub_data_map[T] = t_mi

    if bool(sub_data_map): # if not empty
        split_point = max(sub_data_map, key=sub_data_map.get)
        if split_point not in split_points[attribute]:
            split_points[attribute].append(split_point)
        if attribute not in nodes_info_per_attribute:
            nodes_info_per_attribute[attribute] = []

        for node_tuple in nodes_info_per_attribute[attribute]:
            if node_tuple[0] == 0:
                new_node_tuple = (0, node_tuple[1] + sub_data_map[split_point])
                nodes_info_per_attribute[attribute].append(new_node_tuple)
                nodes_info_per_attribute[attribute].remove(node_tuple)
                break

        sub_interval_0 = []
        sub_interval_1 = []
        for elem in sub_interval:
            if elem[0] < split_point:
                sub_interval_0.append(elem)
            else:
                sub_interval_1.append(elem)

        if len(sub_interval_0) > 0:
            first_loop_recursive_split_points(sub_interval_0, attribute, self)
        if len(sub_interval_1) > 0:
            first_loop_recursive_split_points(sub_interval_1, attribute, self)



def recursive_split_points(sub_interval, attribute, self, nodes,distinct_attribute_data):
    sub_interval_0 = {}
    sub_interval_1 = {}

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

            for data_class_tuple in sub_interval[node.index]:
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
        if split_point not in split_points[attribute]:
            split_points[attribute].append(split_point)
        if attribute not in nodes_info_per_attribute:
            nodes_info_per_attribute[attribute] = []

        new_tuple_list = []
        for node_tuple in nodes_info_per_attribute[attribute]:
            if split_point in node_T_significants[node_tuple[0]].keys():
                new_node_tuple = (node_tuple[0], node_tuple[1] + node_T_significants[node_tuple[0]][split_point])
                new_tuple_list.append(new_node_tuple)
            else:
                new_tuple_list.append(node_tuple)

        nodes_info_per_attribute[attribute] = new_tuple_list

        sub_interval_0 = {}
        sub_interval_1 = {}

        for node_i, values in sub_interval.items():
            sub_interval_0[node_i] = []
            sub_interval_1[node_i] = []
            for elem in values:
                if elem[0] < split_point:
                    sub_interval_0[node_i].append(elem)
                else:
                    sub_interval_1[node_i].append(elem)

    if len(sub_interval_0) > 0:
        recursive_split_points(sub_interval_0, attribute, self, nodes, distinct_attribute_data)
    if len(sub_interval_1) > 0:
        recursive_split_points(sub_interval_1, attribute, self, nodes, distinct_attribute_data)


def find_split_position(record, positions):
    if record < positions[0]:
        return 0

    if record >= positions[len(positions)-1]:
        return len(positions)

    for i in range(len(positions)):
        first_position = positions[i]
        second_position = positions[i+1]
        if first_position <= record < second_position:
            return i+1