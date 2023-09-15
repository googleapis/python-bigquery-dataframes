# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/indexes/base.py

from bigframes import constants


class Index:
    """Immutable sequence used for indexing and alignment.

    The basic object storing axis labels for all objects.
    """

    @property
    def name(self):
        """Returns Index name."""
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def shape(self):
        """
        Return a tuple of the shape of the underlying data.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def nlevels(self) -> int:
        """Number of levels."""
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def is_unique(self) -> bool:
        """Return if the index has unique values."""
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def to_numpy(self, dtype):
        """
        A NumPy ndarray representing the values in this Series or Index.

        Args:
            dtype:
                The dtype to pass to :meth:`numpy.asarray`.
            **kwargs:
                Additional keywords passed through to the ``to_numpy`` method
                of the underlying array (for extension arrays).

        Returns:
            numpy.ndarray
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
