__all__ = [
    '_FeatureInfo'
]

class _FeatureInfo:
    r"""Class for feature representation.
    
    Date:
        2021

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        TODO
    """

    def __init__(self, dtype, values=None, min_val=None, max_val=None, **kwargs):
        r"""Initialize instance of _FeatureInfo.

        Arguments:
            TODO
        """
        self.dtype = dtype
        self.values = values
        self.min = min_val
        self.max = max_val