from pandas.api.types import is_numeric_dtype
from niaclass.feature_info import _FeatureInfo
from niaclass.rule import _Rule
from niapy.task import Task
from niapy.problems import Problem
from niapy.util.factory import get_algorithm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, cohen_kappa_score

__all__ = ["NiaClass", "_NiaClassProblem"]


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
        __history (Iterable[_Rule]): History of all identified rules.
    """

    def __init__(
        self,
        pop_size=90,
        num_evals=5000,
        score_func_name="accuracy",
        algo="FireflyAlgorithm",
        **kwargs
    ):
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
        self.__algo_args["population_size"] = self.__pop_size
        self.__rules = None
        self.__history = []

    def fit(self, x, y):
        r"""Fit NiaClass.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.
            y (pandas.core.series.Series): n classes of the samples in the x array.

        Returns:
            None
        """
        feats = []

        for col in x:
            if is_numeric_dtype(x[col]):
                feats.append(_FeatureInfo(1, None, x[col].min(), x[col].max()))
            else:
                feats.append(_FeatureInfo(0, x[col].unique(), None, None))

        preprocessed_data = self.__preprocess_dataset(x, y, feats)

        algo = get_algorithm(self.__algo)
        algo.set_parameters(**self.__algo_args)

        problem = _NiaClassProblem(
            feats,
            y.unique(),
            x,
            y,
            preprocessed_data,
            self.__score_func_name,
            self.__classify,
        )
        task = Task(problem, max_evals=self.__num_evals)
        algo.run(task)

        self.__rules = problem.get_rules()
        self.__history = problem.get_history()

    def predict(self, x):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            Iterable[any]: n predicted classes.
        """
        if not self.__rules:
            raise Exception(
                "This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )

        return self.__classify(x, self.__rules)

    def __preprocess_dataset(self, x, y, feature_infos):
        r"""Extract useful information from dataset to use for fitting.

        Arguments:
            x (pandas.core.frame.DataFrame): Individuals.
            y (pandas.core.series.Series): Individuals' classes.
            feature_infos (Iterable[_FeatureInfo]): List of _FeatureInfo instances.

        Returns:
            Dict[any, Iterable[Tuple[float, float, float]]]: Preprocessed numeric features for each class.
        """
        data = {}
        classes = y.unique()
        for c in classes:
            data[c] = []
            locations = y[y == c].index
            individuals = x.loc[locations]
            for i in range(len(feature_infos)):
                if feature_infos[i].dtype == 1:  # if numeric feature
                    col = individuals.iloc[:, i]  # get values of feature from x
                    col = col[
                        ~((col - col.mean()).abs() > 3 * col.std())
                    ]  # remove outliers if any
                    max_val = x.iloc[:, i].max()  # get maximum value for normalization
                    if max_val == 0:
                        data[c].append((0, 0, 0))
                    else:
                        norm_min = col.min() / max_val
                        norm_max = col.max() / max_val
                        data[c].append(
                            (norm_min, norm_max, norm_max - norm_min)
                        )  # normalized minimum, maximum and distance
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
                    elif rules[i].min <= individual[i] <= rules[i].max:
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

    def print_best_rules(self):
        """Print the best set of rules found during the optimization process."""
        if not self.__rules:
            print("No rules available.")
            return

        print("Rules identified:")
        for class_label, rules in self.__rules.items():
            print(f"Class: {class_label}")
            for i, rule in enumerate(rules):
                if rule is None:
                    print(f"  Feature {i}: No rule")
                else:
                    if rule.value is not None:
                        print(f"  Feature {i}: Equals {rule.value}")
                    else:
                        print(f"  Feature {i}: Range [{rule.min}, {rule.max}]")

    
    def print_all_rules(self):
        """Print the rules found during the optimization process."""
        if len(self.__history) == 0:
            print("No rules available.")
            return

        print("Rules identified:")
        for iteration in self.__history:
            for class_label, rules in iteration.items():
                print(f"Class: {class_label}")
                for i, rule in enumerate(rules):
                    if rule is None:
                        print(f"  Feature {i}: No rule")
                    else:
                        if rule.value is not None:
                            print(f"  Feature {i}: Equals {rule.value}")
                        else:
                            print(f"  Feature {i}: Range [{rule.min}, {rule.max}]")

class _NiaClassProblem(Problem):
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
        __rules (Iterable[_Rule]): List of all identified rules.
    """

    def __init__(
        self, features, classes, x, y, preprocessed_data, score_func_name, classify_func
    ):
        r"""Initialize instance of _NiaClassProblem.

        Arguments:
            features (Iterable[_FeatureInfo]): List of _FeatureInfo instances.
            classes (Iterable[any]): Unique classes.
            x (pandas.core.frame.DataFrame): Individuals.
            y (pandas.core.series.Series): Individuals' classes.
            preprocessed_data (Dict[str, any]): Dictionary containing preprocessed data for all classes.
            score_func_name (str): Used score function.
            classify_func (Callable[[pandas.core.frame.DataFrame, Iterable[_Rule]], pandas.core.series.Series]): Function for classification.
        """
        num_of_classes = len(y)
        dimension = 1  # 1 for control value that threshold is compared to.
        for f in features:
            if f.dtype == 1:
                """
                * 1 for threshold that determines if the definite feature belongs to the rule or not.
                * 2 for min and max mapping for each class (num_of_classes).
                """
                dimension += 1 + 2 * num_of_classes
            else:
                """
                * 1 for threshold that determines if the definite feature belongs to the rule or not
                """
                dimension += 1 + num_of_classes
        super().__init__(dimension=dimension, lower=0.0, upper=1.0)
        self.__features = features
        self.__classes = classes
        self.__x = x
        self.__y = y
        self.__preprocessed_data = preprocessed_data
        self.__current_best_score = np.inf
        self.__current_best_rules = None
        self.__classify_func = classify_func
        self.__score_func_name = score_func_name
        self.__rules = []

    def get_rules(self):
        r"""Returns current best set of rules.

        Returns:
            Iterable[_Rule]: Best set of rules found during the optimization process.
        """
        return self.__current_best_rules
    
    def get_history(self):
        r"""Returns list of all identified rules.
        
        Returns:
            Iterable[_Rule]: List of all identified rules found during the optimization process.
        """
        return self.__rules

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
        """Calculates the overlapping distance of two intervals [start1, end1] and [start2, end2].

        Arguments:
            start1 (float): Minimum value of the first interval.
            start2 (float): Minimum value of the second interval.
            end1 (float): Maximum value of the first interval.
            end2 (float): Maximum value of the second interval.
        Returns:
            float: Overlap.
        """
        total_range = max([end1, end2]) - min([start1, start2])
        sum_of_ranges = (end1 - start1) + (end2 - start2)
        return (
            min([end1, end2]) - max([start1, start2])
            if sum_of_ranges > total_range
            else 0
        )

    def __build_rules(self, sol):
        """Builds a set of rules for the input solution candidate.

        Arguments:
            sol (Iterable[float]): Solution candidate.

        Returns:
            Tuple[Dict[str, Iterable[_Rule]], float, float]:
                1. Built set of rules for each possible class.
                2. Normalized sum of all interval differences.
                3. Normalized sum of overlapping distances.
        """
        classes_rules = {k: [] for k in self.__classes}
        interval_lengths = []
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
                        length = val2_norm - val1_norm
                        (
                            val1_preprocessing,
                            val2_preprocessing,
                            length_preprocessing,
                        ) = self.__preprocessed_data[k][num_feature_index]
                        overlapping = self.__overlapping_range(
                            val1_norm, val1_preprocessing, val2_norm, val2_preprocessing
                        )

                        # difference in lengths of randomly selected interval and interval calculated from the training set (both normalized)
                        interval_lengths.append(length - length_preprocessing)
                        # overlapping distance of both (normalized) intervals
                        overlaps.append(overlapping)

                        classes_rules[k].append(_Rule(None, val1, val2))
                        sol_ind += 2
                    else:
                        classes_rules[k].append(
                            _Rule(
                                f.values[
                                    self.__get_bin_index(sol[sol_ind], len(f.values))
                                ],
                                None,
                                None,
                            )
                        )
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
        if not np.any(
            np.array(classes_rules[list(classes_rules.keys())[0]], dtype=object)
        ):
            return None, None, None

        # rules,
        # normalized sum of all interval differences (for all numeric features in all classes) - we assume that it is better if difference among intervals is smaller, hence we add this value multiplied by some weight in the fitness function,
        # normalized sum of overlapping distances (for all numeric features in all classes) - we assume that it is better if overlapping distance in bigger, hende we subtract this value multiplied by some weight in the fitness function
        # WE ARE DEALING WITH MINIMIZATION PROBLEM
        return (
            classes_rules,
            (np.sum(interval_lengths) / len(interval_lengths))
            if len(interval_lengths) > 0
            else 0,
            (np.sum(overlaps) / len(overlaps)) if len(overlaps) > 0 else 0,
        )

    def __score(self, score_name, y_predicted):
        """Calculates the score, using the specified score function's name, of predicted classes.

        Arguments:
            score_name (str): Score function's name.
            y_predicted (pandas.core.series.Series): Predicted classes.

        Returns:
            float: Calculated score.
        """
        if score_name == "accuracy":
            return accuracy_score(self.__y, y_predicted)
        elif score_name == "precision":
            return precision_score(self.__y, y_predicted, average="weighted")
        elif score_name == "f1":
            return f1_score(self.__y, y_predicted, average="weighted")
        elif score_name == "cohen_kappa":
            return cohen_kappa_score(self.__y, y_predicted)
        else:
            raise Exception("Score function not implemented.")

    def _evaluate(self, x):
        r"""Override fitness function.

        Args:
            x (numpy.ndarray): Solution vector.

        Returns:
            float: Fitness value of `x`
        """

        classes_rules, length_diffs, overlaps = self.__build_rules(x)
        if not classes_rules:
            return np.inf

        y = self.__classify_func(self.__x, classes_rules)

        # score = -self.__score(self.__score_func_name, y)
        score = (
            -self.__score(self.__score_func_name, y)
            + 0.5 * length_diffs
            - 0.5 * overlaps
        )
        self.__rules.append(classes_rules)
        if score < self.__current_best_score:
            self.__current_best_score = score
            self.__current_best_rules = classes_rules

        return score
