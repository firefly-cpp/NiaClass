from pandas.api.types import is_numeric_dtype
from niaclass.feature_info import _FeatureInfo
from niaclass.rule import _Rule
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Benchmark
from NiaPy.algorithms.utility import AlgorithmUtility

__all__ = [
    'NiaClass'
]

class NiaClass:
    r"""Implementation of NiaClass classifier.
    
    Date:
        2021

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        TODO
    """

    def __init__(self, pop_size=90, num_evals=5000, algo='FireflyAlgorithm', **kwargs):
        r"""Initialize instance of NiaClass.

        Arguments:
            TODO
        """
        self.__pop_size = pop_size
        self.__num_evals = num_evals
        self.__algo = algo
    
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
            if f.dtype is 1:
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

        benchmark = _NiaClassBenchmark()
        task = StoppingTask(
            D=D,
            nFES=self.__num_evals,
            benchmark=benchmark
        )
        algo.run(task)

    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return None

class _NiaClassBenchmark(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, 0.0, 1.0)

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D): val += sol[i] ** 2
            return val
        return evaluate