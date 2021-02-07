from pandas.api.types import is_numeric_dtype
from pandas import Series
from niaclass.feature_info import _FeatureInfo
from niaclass.rule import _Rule
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Benchmark
from NiaPy.algorithms.utility import AlgorithmUtility
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, cohen_kappa_score

__all__ = [
    'NiaClass',
    '_NiaClassBenchmark'
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
        __score_func_name (Optional(str)): Used score function.
        __algo (str): Name of the optimization algorithm to use.
        __rules (Dict[any, Iterable[_Rule]]): Best set of rules found in the optimization process.
    """

    def __init__(self, pop_size=90, num_evals=5000, score_func_name='accuracy', algo='FireflyAlgorithm', **kwargs):
        r"""Initialize instance of NiaClass.

        Arguments:
            pop_size (Optional(int)): Number of individuals in the fitting process.
            num_evals (Optional(int)): Maximum evaluations in the fitting process.
            score_func_name (Optional(str)): Used score function.
            algo (Optional(str)): Name of the optimization algorithm to use.
        """
        self.__pop_size = pop_size
        self.__num_evals = num_evals
        self.__score_func_name = score_func_name
        self.__algo = algo
        self.__algo_args = kwargs
        self.__algo_args['NP'] = self.__pop_size
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
        
        preprocessed_data = self.__preprocess_dataset(x, y, feats)

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
        algo.setParameters(**self.__algo_args)

        benchmark = _NiaClassBenchmark(feats, y.unique(), x, y, preprocessed_data, self.__score_func_name, self.__classify)
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
            Iterable[any]: n predicted classes.
        """
        if not self.__rules:
            raise Exception('This instance is not fitter yet. Call \'fit\' with appropriate arguments before using this estimator.')

        return self.__classify(x, self.__rules)

    def __preprocess_dataset(self, x, y, feature_infos):
        r"""Extract useful information from dataset to use for fitting.

        Arguments:
            x (pandas.core.frame.DataFrame): Individuals.
            y (pandas.core.series.Series): Individuals' classes.
            features (Iterable[_FeatureInfo]): List of _FeatureInfo instances.

        Returns:
            TODO
        """
        data = {}
        classes = y.unique()
        for c in classes:
            data[c] = []
            locations = y[y == c].index
            individuals = x.loc[locations]
            for i in range(len(feature_infos)):
                if feature_infos[i].dtype == 1:
                    col = individuals.iloc[:, i]
                    col = col[~((col - col.mean()).abs() > 3 * col.std())]
                    max_val = x.iloc[:, i].max()
                    if max_val == 0:
                        data[c].append((0, 0, 0))
                    else:
                        norm_min = col.min() / max_val
                        norm_max = col.max() / max_val
                        data[c].append((norm_min, norm_max, norm_max - norm_min))
        return data
    
    def __classify(self, x, rules):
        r"""Execute classification of individuals for the given rules.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            rules (Iterable[_Rule]): Classification rules.

        Returns:
            Iterable[any]: n predicted classes.
        """
        def __get_class_score(rules, individual):
            r"""Calculate individual's score for the given set of rules.

            Arguments:
                rules (Iterable[_Rule]): Classification rules.
                individual (pandas.core.series.Series): List of an individual's features.

            Returns:
                float: Individual's score.
            """
            score = 0
            for i in range(len(individual)):
                if rules[i] is not None:
                    if rules[i].value:
                        if individual[i] == rules[i].value:
                            score += 1
                    elif individual[i] >= rules[i].min and individual[i] <= rules[i].max:
                        score += 1
            return score

        y = []
        for i, row in x.iterrows():
            current_score = -1
            current_class = None
            for k in rules:
                score = __get_class_score(rules[k], row)
                if score > current_score:
                    current_score = score
                    current_class = k
            y.append(current_class)
        return y

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
        __x (pandas.core.frame.DataFrame): Individuals.
        __y (pandas.core.series.Series): Individuals' classes.
        __preprocessed_data (Dict[str, any]): Dictionary containing preprocessed data for all classes.
        __current_best_score (float): Current best score during optimization.
        __current_best_rules (Dict[any, Iterable[_Rule]]): Dictionary for mapping classes to their rules.
        __score_func_name (str): Used score function.
        __classify_func (Callable[[pandas.core.frame.DataFrame, Iterable[_Rule]], pandas.core.series.Series]): Function for classification.
    """
    def __init__(self, features, classes, x, y, preprocessed_data, score_func_name, classify_func):
        r"""Initialize instance of _NiaClassBenchmark.

        Arguments:
            features (Iterable[_FeatureInfo]): List of _FeatureInfo instances.
            classes (Iterable[any]): Unique classes.
            x (pandas.core.frame.DataFrame): Individuals.
            y (pandas.core.series.Series): Individuals' classes.
            preprocessed_data (Dict[str, any]): Dictionary containing preprocessed data for all classes.
            score_func_name (str): Used score function.
            classify_func (Callable[[pandas.core.frame.DataFrame, Iterable[_Rule]], pandas.core.series.Series]): Function for classification.
        """
        Benchmark.__init__(self, 0.0, 1.0)
        self.__features = features
        self.__classes = classes
        self.__x = x
        self.__y = y
        self.__preprocessed_data = preprocessed_data
        self.__current_best_score = float('inf')
        self.__current_best_rules = None
        self.__classify_func = classify_func
        self.__score_func_name = score_func_name
    
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
    
    def __overlapping_range(self, start1, start2, end1, end2):
        """TODO
        """
        total_range = max([end1, end2]) - min([start1, start2])
        sum_of_ranges = (end1 - start1) + (end2 - start2)
        return min([end1, end2]) - max([start1, start2]) if sum_of_ranges > total_range else 0
    
    def __build_rules(self, sol):
        """Builds a set of rules for the input solution candidate.

        Arguments:
            sol (Iterable[float]): Solution candidate.
        
        Returns:
            Dict[str, Iterable[_Rule]]: Built set of rules for each possible class.
        """
        classes_rules = {k: [] for k in self.__classes}
        interval_distances = []
        overlaps = []
        
        sol_ind = 0
        num_feature_index = 0
        for f in self.__features:
            current_feature_threshold = sol[sol_ind]
            sol_ind += 1
            for k in classes_rules:
                if current_feature_threshold >= sol[-1]:
                    if f.dtype == 1:
                        val1 = sol[sol_ind] * f.max + f.min
                        val2 = sol[sol_ind + 1] * f.max + f.min
                        
                        val1 = f.max if val1 > f.max else val1
                        val2 = f.max if val2 > f.max else val2
                        
                        (val1, val2) = (val2, val1) if val2 < val1 else (val1, val2)
                        (val1_norm, val2_norm) = (val1 / f.max, val2 / f.max)
                        distance = val2_norm - val1_norm
                        (val1_preprocessing, val2_preprocessing, distance_preprocessing) = self.__preprocessed_data[k][num_feature_index]
                        overlapping = self.__overlapping_range(val1_norm, val1_preprocessing, val2_norm, val2_preprocessing)

                        interval_distances.append(distance - distance_preprocessing)
                        overlaps.append(overlapping)

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
            if f.dtype == 1:
                num_feature_index += 1

        # if all rules are None
        if not np.any(np.array(classes_rules[list(classes_rules.keys())[0]], dtype=object)): return None, None, None

        return classes_rules, (np.sum(interval_distances) / len(interval_distances)) if len(interval_distances) > 0 else 0, (np.sum(overlaps) / len(overlaps)) if len(overlaps) > 0 else 0
    
    def __score(self, score_name, y_predicted):
        """Calculates the score, using the specified score function's name, of predicted classes.

        Arguments:
            score_name (str): Score function's name.
            y_predicted (pandas.core.series.Series): Predicted classes.
        
        Returns:
            float: Calculated score.
        """
        if score_name == 'accuracy':
            return accuracy_score(self.__y, y_predicted)
        elif score_name == 'precision':
            return precision_score(self.__y, y_predicted, average='weighted')
        elif score_name == 'f1':
            return f1_score(self.__y, y_predicted, average='weighted')
        elif score_name == 'cohen_kappa':
            return cohen_kappa_score(self.__y, y_predicted)
        else:
            raise Exception('Score function not implemented.')

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
            classes_rules, distance_factor, overlaps = self.__build_rules(sol)
            if not classes_rules: return float('inf')

            y = self.__classify_func(self.__x, classes_rules)

            #score = -self.__score(self.__score_func_name, y)
            #score = -self.__score(self.__score_func_name, y) - 0.5 * overlaps
            #score = -self.__score(self.__score_func_name, y) + 0.5 * distance_factor - 0.5 * overlaps
            score = -self.__score(self.__score_func_name, y) + 0.5 * distance_factor
            if score < self.__current_best_score:
                self.__current_best_score = score
                self.__current_best_rules = classes_rules
            
            return score
        return evaluate
