from pandas.api.types import is_numeric_dtype
from pandas import Series
from niaclass.feature_info import _FeatureInfo
from niaclass.rule import _Rule
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Benchmark
from NiaPy.algorithms.utility import AlgorithmUtility
import numpy as np

__all__ = [
    'NiaClass'
]

class NiaClass:
    r"""Implementation of NiaClass classifier.
    
    Date:
        2021

    Author:
        Luka Pečnik

    Reference:
        The implementation is based on the following article:
        Iztok Fister Jr., Iztok Fister, Dušan Fister, Grega Vrbančič, Vili Podgorelec. On the potential of the nature-inspired algorithms for pure binary classification. In. Computational science - ICCS 2020 : 20th International Conference, Proceedings. Part V. Cham: Springer, pp. 18-28. Lecture notes in computer science, 12141, 2020

    License:
        MIT

    Attributes:
        __pop_size (int): Number of individuals in the fitting process.
        __num_evals (int): Maximum evaluations in the fitting process.
        __algo (str): Name of the optimization algorithm to use.
        __rules (Dict[any, Iterable[_Rule]]): Best set of rules found in the optimization process.
    """

    def __init__(self, pop_size=90, num_evals=5000, algo='FireflyAlgorithm', **kwargs):
        r"""Initialize instance of NiaClass.

        Arguments:
            pop_size (Optional(int)): Number of individuals in the fitting process.
            num_evals (Optional(int)): Maximum evaluations in the fitting process.
            algo (Optional(str)): Name of the optimization algorithm to use.
        """
        self.__pop_size = pop_size
        self.__num_evals = num_evals
        self.__algo = algo
        self.__rules = None
    
    def fit(self, x, y):
        r"""Fit NiaClass.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        num_of_classes = y.nunique()
        feats = []

        for col in x:
            if is_numeric_dtype(x[col]):
                feats.append(_FeatureInfo(1, None, x[col].min(), x[col].max()))
            else:
                feats.append(_FeatureInfo(0, x[col].unique(), None, None))
        
        D = 1 # 1 for control value that threshold is compared to.
        for f in feats:
            if f.dtype == 1:
                """
                * 1 for threshold that determines if the definite feature belongs to the rule or not.
                * 2 for min and max mapping for each class (num_of_classes).
                """
                D += 1 + 2 * num_of_classes
            else:
                """
                * 1 for threshold that determines if the definite feature belongs to the rule or not
                """
                D += 1 + num_of_classes
        
        algo = AlgorithmUtility().get_algorithm(self.__algo)
        algo.NP = self.__pop_size

        benchmark = _NiaClassBenchmark(feats, y.unique(), x, y, self.__get_class_score)
        task = StoppingTask(
            D=D,
            nFES=self.__num_evals,
            benchmark=benchmark
        )
        algo.run(task)

        self.__rules = benchmark.get_rules()

    def predict(self, x):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            array: n predicted classes.
        """
        if not self.__rules:
            raise Exception('This instance is not fitter yet. Call \'fit\' with appropriate arguments before using this estimator.')

        y = []
        for i, row in x.iterrows():
            current_score = -1
            current_class = None
            for k in self.__rules:
                score = self.__get_class_score(self.__rules[k], row)
                if score > current_score:
                    current_score = score
                    current_class = k
            y.append(current_class)
        return y
    
    def __get_class_score(self, rules, individual):
        r"""Calculate individual's score for the given set of rules.

        Arguments:
            rules (Iterable[_Rule]): n samples to classify.
            individual (pandas.core.series.Series): List of an individual's features.

        Returns:
            float: Individual's score.
        """
        score = 0
        for i in range(len(individual)):
            if rules[i] is not None:
                if rules[i].value is not None and individual[i] == rules[i].value:
                    score += 1
                elif individual[i] >= rules[i].min and individual[i] <= rules[i].max:
                    score += 1
        return score

class _NiaClassBenchmark(Benchmark):
    r"""Implementation of Benchmark class from NiaPy library.

    Date:
        2021

    Author
        Luka Pečnik

    License:
        MIT

    Attributes:
        __features (Iterable[_FeatureInfo]): List of _FeatureInfo instances.
        __classes (Iterable[any]): Unique classes.
        __x (pandas.core.frame.DataFrame): individuals.
        __y (pandas.core.series.Series): individuals' classes.
        __current_best_score (float): current best score during optimization.
        __current_best_rules (Dict[any, Iterable[_Rule]]): dictionary for mapping classes to their rules.
        __get_class_func (Callable[[Iterable[_Rule], pandas.core.series.Series], float]): function for calculating individual's score for class.
    """
    def __init__(self, features, classes, x, y, get_class_func):
        r"""Initialize instance of _NiaClassBenchmark.

        Arguments:
            features (Iterable[_FeatureInfo]): List of _FeatureInfo instances.
            classes (Iterable[any]): Unique classes.
            x (pandas.core.frame.DataFrame): individuals.
            y (pandas.core.series.Series): individuals' classes.
            get_class_func (Callable[[Iterable[_Rule], pandas.core.series.Series], float]): function for calculating individual's score for class.
        """
        Benchmark.__init__(self, 0.0, 1.0)
        self.__features = features
        self.__classes = classes
        self.__x = x
        self.__y = y
        self.__current_best_score = float('inf')
        self.__current_best_rules = None
        self.__get_class_func = get_class_func
    
    def get_rules(self):
        r"""Returns current best set of rules.

        Returns:
            Iterable[_Rule]: Best set of rules found during the optimization process.
        """
        return self.__current_best_rules

    def __get_bin_index(self, value, number_of_bins):
        """Gets index of value's bin. Value must be between 0.0 and 1.0.

        Arguments:
            value (float): Value to put into bin.
            number_of_bins (uint): Number of bins on the interval [0.0, 1.0].
        
        Returns:
            uint: Calculated index.
        """
        bin_index = np.int(np.floor(value / (1.0 / number_of_bins)))
        if bin_index >= number_of_bins:
            bin_index -= 1
        return bin_index
    
    def __build_rules(self, sol):
        """Builds a set of rules for the input solution candidate.

        Arguments:
            sol (Iterable[float]): Solution candidate.
        
        Returns:
            Dict[str, Iterable[_Rule]]: Built set of rules for each possible class.
        """
        classes_rules = {k: [] for k in self.__classes}
        
        sol_ind = 0
        for f in self.__features:
            current_feature_threshold = sol[sol_ind]
            sol_ind += 1
            for k in classes_rules:
                if current_feature_threshold >= sol[-1]:
                    if f.dtype == 1:
                        val1 = sol[sol_ind] * f.max + f.min
                        val2 = sol[sol_ind + 1] * f.max + f.min
                        (val1, val2) = (val2, val1) if val2 < val1 else (val1, val2)

                        classes_rules[k].append(_Rule(None, val1, val2))
                        sol_ind += 2
                    else:
                        classes_rules[k].append(_Rule(f.values[self.__get_bin_index(sol[sol_ind], len(f.values))], None, None))
                        sol_ind += 1
                else:
                    if f.dtype == 1:
                        sol_ind += 2
                    else:
                        sol_ind += 1
                    classes_rules[k].append(None)

        # if all rules are None
        if not np.any(np.array(classes_rules[list(classes_rules.keys())[0]], dtype=object)): return None

        return classes_rules
    
    def __accuracy(self, y_predicted):
        """Calculates accuracy of predicted classes.

        Arguments:
            y_predicted (pandas.core.series.Series): Predicted classes.
        
        Returns:
            float: Calculated accuracy.
        """
        matches = self.__y.shape[0] - self.__y.compare(y_predicted).shape[0]
        return matches / self.__y.shape[0]

    def function(self):
        r"""Override Benchmark function.

        Returns:
            Callable[[int, numpy.ndarray[float]], float]: Fitness evaluation function.
        """
        def evaluate(D, sol):
            r"""Evaluate solution.

            Arguments:
                D (uint): Number of dimensionas.
                sol (numpy.ndarray[float]): Individual of population/ possible solution.

            Returns:
                float: Fitness.
            """
            classes_rules = self.__build_rules(sol)
            if not classes_rules: return float('inf')

            y = []
            for i, row in self.__x.iterrows():
                current_score = -1
                current_class = None
                for k in classes_rules:
                    score = self.__get_class_func(classes_rules[k], row)
                    if score > current_score:
                        current_score = score
                        current_class = k
                y.append(current_class)
            y = Series(y)

            accuracy = -self.__accuracy(y)
            if accuracy < self.__current_best_score:
                self.__current_best_score = accuracy
                self.__current_best_rules = classes_rules
            
            return accuracy
        return evaluate
