# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/arrays/arrow/accessors.py
"""Accessors for arrow-backed data."""

from __future__ import annotations

from bigframes import constants


class StructAccessor:
    """
    Accessor object for structured data properties of the Series values.
    """

    def field(self, name_or_index: str | int):
        """
        Extract a child field of a struct as a Series.

        Parameters
        ----------
        name_or_index : str | int
            Name or index of the child field to extract.

        Returns
        -------
        pandas.Series
            The data corresponding to the selected child field.

        See Also
        --------
        Series.struct.explode : Return all child fields as a DataFrame.

        Examples
        --------
        >>> import bigframes.pandas as bpd
        >>> import pyarrow as pa
        >>> s = bpd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=bpd.ArrowDtype(pa.struct(
        ...         [("version", pa.int64()), ("project", pa.string())]
        ...     ))
        ... )

        Extract by field name.

        >>> s.struct.field("project")
        0    pandas
        1    pandas
        2     numpy
        Name: project, dtype: string[pyarrow]

        Extract by field index.

        >>> s.struct.field(0)
        0    1
        1    2
        2    1
        Name: version, dtype: int64[pyarrow]
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def explode(self):
        """
        Extract all child fields of a struct as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            The data corresponding to all child fields.

        See Also
        --------
        Series.struct.field : Return a single child field as a Series.

        Examples
        --------
        >>> import bigframes.pandas as bpd
        >>> import pyarrow as pa
        >>> s = bpd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=bpd.ArrowDtype(pa.struct(
        ...         [("version", pa.int64()), ("project", pa.string())]
        ...     ))
        ... )

        >>> s.struct.explode()
           version project
        0        1  pandas
        1        2  pandas
        2        1   numpy
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
