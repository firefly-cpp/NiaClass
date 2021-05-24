__all__ = ["_FeatureInfo"]


class _FeatureInfo:
    r"""Class for feature representation.

    Date:
        2021

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        dtype (int): Type of feature.
        values (Optional(Iterable[any])): Possible categorical feature's values.
        min (Optional(float)): Maximum numerical feature's value.
        max (Optional(float)): Minimum numerical feature's value.
    """

    def __init__(self, dtype, values=None, min_val=None, max_val=None, **kwargs):
        r"""Initialize instance of _FeatureInfo.

        Arguments:
            dtype (int): Type of feature.
            values (Optional(Iterable[any])): Possible categorical feature's values.
            min (Optional(float)): Maximum numerical feature's value.
            max (Optional(float)): Minimum numerical feature's value.
        """
        self.dtype = dtype
        self.values = values
        self.min = min_val
        self.max = max_val
