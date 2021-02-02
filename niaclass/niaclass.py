from pandas.api.types import is_numeric_dtype

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
        TODO

    Attributes:
        TODO
    """

    def __init__(self, **kwargs):
        r"""Initialize instance of NiaClass.
        """
    
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

        return None
    
    def predict(self, x, **kwargs):
        r"""Predict class for each sample (row) in x.

        Arguments:
            x (pandas.core.frame.DataFrame): n samples to classify.

        Returns:
            pandas.core.series.Series: n predicted classes.
        """
        return None

class _FeatureInfo:
    r"""Class for feature representation.
    
    Date:
        2021

    Author:
        Luka Pečnik

    License:
        TODO

    Attributes:
        TODO
    """

    def __init__(self, dtype, values = None, min_val = None, max_val = None, **kwargs):
        r"""Initialize instance of _FeatureInfo.

        Arguments:
            TODO
        """
        self.dtype = dtype
        self.values = values
        self.min = min_val
        self.max = max_val