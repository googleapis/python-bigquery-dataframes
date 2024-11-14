# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/indexes/base.py
from __future__ import annotations

import typing

from bigframes import constants


class Index:
    """Immutable sequence used for indexing and alignment.

    The basic object storing axis labels for all objects.

    Args:
        data (pandas.Series | pandas.Index | bigframes.series.Series | bigframes.core.indexes.base.Index):
            Labels (1-dimensional).
        dtype:
            Data type for the output Index. If not specified, this will be
            inferred from `data`.
        name:
            Name to be stored in the index.
        session (Optional[bigframes.session.Session]):
            BigQuery DataFrames session where queries are run. If not set,
            a default session is used.
    """

    @property
    def name(self):
        """Returns Index name.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([1, 2, 3], name='x')
            Index([1, 2, 3], dtype='Int64', name='x')
            >>> idx.name
            'x'

        Returns:
            blocks.Label:
                Index or MultiIndex name
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def values(self):
        """Return an array representing the data in the Index.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([1, 2, 3])
            >>> idx
            Index([1, 2, 3], dtype='Int64')

            >>> idx.values
            array([1, 2, 3])

        Returns:
            array:
                Numpy.ndarray or ExtensionArray
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def shape(self):
        """
        Return a tuple of the shape of the underlying data.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = pd.Index([1, 2, 3])
            >>> idx
            Index([1, 2, 3], dtype='Int64')

            >>> idx.shape
            (3,)

            Returns:
                Tuple[int]:
                    A tuple of int representing the shape.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def nlevels(self) -> int:
        """Integer number of levels in this MultiIndex

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> mi = bpd.MultiIndex.from_arrays([['a'], ['b'], ['c']])
            >>> mi
            MultiIndex([('a', 'b', 'c')],
           ...         )
           >>> mi.nlevels
           3

        Returns:
            Int:
                Number of levels.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def is_unique(self) -> bool:
        """Return if the index has unique values.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([1, 5, 7, 7])
            >>> idx.is_unique
            False

            >>> idx = bpd.Index([1, 5, 7])
            >>> idx.is_unique
            True

        Returns:
            bool:
                True if the index has unique values, otherwise False.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def has_duplicates(self) -> bool:
        """Check if the Index has duplicate values.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([1, 5, 7, 7])
            >>> idx.has_duplicates
            True

            >>> idx = bpd.Index([1, 5, 7])
            >>> idx.has_duplicates
            False

            >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
                ...             "Watermelon"]).astype("category")
            >>> idx.has_duplicates
            True

            >>> idx = pd.Index(["Orange", "Apple",
                ...             "Watermelon"]).astype("category")
            >>> idx.has_duplicates
            False

        Returns:
            bool:
                True if the Index has duplicate values, otherwise False.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def dtype(self):
        """Return the dtype object of the underlying data.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([1, 2, 3])
            >>> idx
            Index([1, 2, 3], dtype='Int64')

            >>> idx.dtype
            Int64Dtype()
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def dtypes(self):
        """Return the dtypes as a Series for the underlying MultiIndex.

        Returns:
            Pandas.Series:
                Pandas.Series of the MultiIndex dtypes.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def T(self) -> Index:
        """Return the transpose, which is by definition self.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> s = bpd.Series(['Ant', 'Bear', 'Cow'])
            >>> s
            0     Ant
            1    Bear
            2     Cow
            dtype: string

            >>> s.T

        For Index:

            >>> idx = bpd.Index([1, 2, 3])
            >>> idx.T
            Index([1, 2, 3], dtype='Int64')

        Returns:
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def copy(
        self,
        name=None,
    ) -> Index:
        """
        Make a copy of this object.

        Name is set on the new object.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = pd.Index(['a', 'b', 'c'])
            >>> new_idx = idx.copy()
            >>> idx is new_idx
            False

        Args:
            name (Label, optional):
                Set name for new object.

        Returns:
            Index:
                Index reference to new object, which is a copy of this object.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def transpose(self) -> Index:
        """
        Return the transpose, which is by definition self.

        Returns:
            bigframes.pandas.Index
                An Index.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def astype(self, dtype):
        """Create an Index with values cast to dtypes.

        The class of a new Index is determined by dtype. When conversion is
        impossible, a TypeError exception is raised.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = pd.Index([1, 2, 3])
            >>> idx
            Index([1, 2, 3], dtype='Int64')


        Args:
            dtype (numpy dtype or pandas type):

        Returns:
            Index: Index with values cast to specified dtype.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def get_level_values(self, level) -> Index:
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatibility.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index(list('abc'))
            >>> idx
            Index(['a', 'b', 'c'], dtype='string')

            >>> idx.get_level_values(0)
            Index(['a', 'b', 'c'], dtype='string')

        Args:
            level (int or str):
                It is either the integer position or the name of the level.

        Returns:
            Index:
                Calling object, as there is only one level in the Index.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def to_series(self):
        """
        Create a Series with both index and values equal to the index keys.

        Useful with map for returning an indexer based on an index.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')

            By default, the original index and original name is reused.

            >>> idx.to_series()
            animal
            Ant      Ant
            Bear    Bear
            Cow      Cow
            Name: animal, dtype: string

            To enforce a new index, specify new labels to index:

            >>> idx.to_series(index=[0, 1, 2])
            0     Ant
            1    Bear
            2     Cow
            Name: animal, dtype: string

            To override the name of the resulting column, specify name:

            >>> idx.to_series(name='zoo')
            animal
            Ant      Ant
            Bear    Bear
            Cow      Cow
            Name: zoo, dtype: string

        Args:
            index (Index, optional):
                Index of resulting Series. If None, defaults to original index.
            name (str, optional):
                Name of resulting Series. If None, defaults to name of original
                index.

        Returns:
            bigframes.pandas.Series:
                The dtype will be based on the type of the Index values.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def isin(self, values):
        """
        Return a boolean array where the index values are in `values`.

        Compute boolean array to check whether each index value is found in the
        passed set of values. The length of the returned boolean array matches
        the length of the index.

        Args:
            values (set or list-like):
                Sought values.

        Returns:
            Series: Series of boolean values.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def all(self) -> bool:
        """Return whether all elements are Truthy.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            True, because nonzero integers are considered True.

            >>> bpd.Index([1, 2, 3]).all()
            True

            False, because 0 is considered False.

            >>> bpd.Index([0, 1, 2]).all()
            False

        Args:

        Returns:
            bool:
                A single element array-like may be converted to bool.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def any(self) -> bool:
        """Return whether any element is Truthy.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

            >>> index = bpd.Index([0, 1, 2])
            >>> index.any()
            True

            >>> index = bpd.Index([0, 0, 0])
            >>> index.any()
            False

        Returns:
            bool:
                A single element array-like may be converted to bool.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def min(self):
        """Return the minimum value of the Index.

        Returns:
            scalar: Minimum value.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def max(self):
        """Return the maximum value of the Index.

        Returns:
            scalar: Maximum value.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def argmin(self) -> int:
        """
        Return int position of the smallest value in the series.

        If the minimum is achieved in multiple locations,
        the first row position is returned.

        Returns:
            int: Row position of the minimum value.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def argmax(self) -> int:
        """
        Return int position of the largest value in the Series.

        If the maximum is achieved in multiple locations,
        the first row position is returned.

        Returns:
            int: Row position of the maximum value.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def nunique(self) -> int:
        """Return number of unique elements in the object.

        Excludes NA values by default.

        Returns:
            int
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def sort_values(
        self, *, ascending: bool = True, na_position: str = "last"
    ) -> Index:
        """
        Return a sorted copy of the index.

        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Args:
            ascending (bool, default True):
                Should the index values be sorted in an ascending order.
            na_position ({'first' or 'last'}, default 'last'):
                Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
                the end.

        Returns:
            pandas.Index: Sorted copy of the index.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def value_counts(
        self,
        normalize: bool = True,
        sort: bool = True,
        ascending: bool = False,
        *,
        dropna: bool = True,
    ):
        """Return a Series containing counts of unique values.

        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Args:
            normalize (bool, default False):
                If True, then the object returned will contain the relative
                frequencies of the unique values.
            sort (bool, default True):
                Sort by frequencies.
            ascending (bool, default False):
                Sort in ascending order.
            dropna (bool, default True):
                Don't include counts of NaN.

        Returns:
            Series
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def fillna(self, value) -> Index:
        """
        Fill NA/NaN values with the specified value.

        Args:
            value (scalar):
                Scalar value to use to fill holes (e.g. 0).
                This value cannot be a list-likes.

        Returns:
            Index
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def rename(self, name) -> Index:
        """
        Alter Index or MultiIndex name.

        Able to set new names without level. Defaults to returning new index.
        Length of names must match number of levels in MultiIndex.

        Args:
            name (label or list of labels):
                Name(s) to set.

        Returns:
            Index: The same type as the caller.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def drop(self, labels) -> Index:
        """
        Make new Index with passed list of labels deleted.

        Args:
            labels (array-like or scalar):

        Returns:
            Index: Will be same type as self.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def dropna(self, how: typing.Literal["all", "any"] = "any"):
        """Return Index without NA/NaN values.

        Args:
            how ({'any', 'all'}, default 'any'):
                If the Index is a MultiIndex, drop the value when any or all levels
                are NaN.

        Returns:
            Index
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def drop_duplicates(self, *, keep: str = "first"):
        """
        Return Index with duplicate values removed.

        Args:
            keep ({'first', 'last', ``False``}, default 'first'):
                One of:
                'first' : Drop duplicates except for the first occurrence.
                'last' : Drop duplicates except for the last occurrence.
                ``False`` : Drop all duplicates.

        Returns:
            Index
        """
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
