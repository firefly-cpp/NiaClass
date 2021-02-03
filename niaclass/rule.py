__all__ = [
    '_Rule'
]

class _Rule:
    r"""Class for rule representation.
    
    Date:
        2021

    Author:
        Luka Pečnik

    License:
        MIT

    Attributes:
        TODO
    """

    def __init__(self, value=None, min_val=None, max_val=None, **kwargs):
        r"""Initialize instance of _Rule.

        Arguments:
            TODO
        """
        self.value = value
        self.min = min_val
        self.max = max_val