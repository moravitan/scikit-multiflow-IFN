import math
import scipy.stats as stats
import numpy as np



class MetaLearning:

    def __init__(self, alpha, number_of_classes):
        if 0 <= alpha < 1:
            self._alpha = alpha
        else:
            raise ValueError("Enter a valid alpha between 0 to 1")
        if 1 < number_of_classes:
            self._classes = number_of_classes
        else:
            raise ValueError("Enter number of classes bigger than 1")
        self._window = 0

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        self._window = value

    def calculate_Wint(self, Pe):

        chi2_alpha = stats.chi2.ppf(self.alpha, self.classes - 1)
        entropy_Pe = stats.entropy([Pe, 1 - Pe], base=2)

        denominator = 2 * np.log(2) * \
                      (math.log(self.classes, 2) - entropy_Pe - Pe * math.log(self.classes - 1, 2))

        if denominator == 0:
            self.window = 0
        else:
            self.window = int(chi2_alpha / denominator)

        return self.window

    def calculate_new_window(self, NI, T, Etr):

        chi2_alpha = stats.chi2.ppf(self.alpha, (NI - 1) * (self.classes - 1))
        entropy_T = stats.entropy([T, 1 - T], base=2)
        entropy_Etr = stats.entropy([Etr, 1 - Etr], base=2)

        denominator = 2 * np.log(2) * \
                      (entropy_T - entropy_Etr - Etr * math.log(self.classes - 1, 2))

        if denominator == 0:
            self.window = 0
        else:
            self.window = int(chi2_alpha / denominator)

        return self.window

    def _calculate_var_diff(self, Etr, Eval, add_count):

        return (Etr * (1 - Etr) / self.window) + (Eval * (1 - Eval) / add_count)

    def get_max_diff(self, Etr, Eval, add_count):

        return stats.norm.ppf(0.99) * self._calculate_var_diff(Etr, Eval, add_count)
