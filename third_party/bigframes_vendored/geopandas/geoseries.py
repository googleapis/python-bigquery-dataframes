from __future__ import annotations

from typing import TYPE_CHECKING

from bigframes import constants

if TYPE_CHECKING:
    import bigframes.series


class GeoSeries:
    """
    A Series object designed to store shapely geometry objects.
    """

    @property
    def x(self) -> bigframes.series.Series:
        """Return the x location of point geometries in a GeoSeries

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None
            >>> from shapely.geometry import Point
            >>> import geopandas

            >>> series = bigframes.pandas.Series(
            ...     [shapely.Point(1, 1), shapely.Point(2, 2), shapely.Point(3, 3)],
            ...     dtype=geopandas.array.GeometryDtype()
            ... )
            >>> s.x
            0    1.0
            1    2.0
            2    3.0
            dtype: float64

        Returns:
            bigframes.series.Series:
                Return the x location (longitude) of point geometries.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def y(self) -> bigframes.series.Series:
        """Return the y location of point geometries in a GeoSeries

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None
            >>> from shapely.geometry import Point
            >>> import geopandas

            >>> series = bigframes.pandas.Series(
            ...     [shapely.Point(1, 1), shapely.Point(2, 2), shapely.Point(3, 3)],
            ...     dtype=geopandas.array.GeometryDtype()
            ... )
            >>> s.y
            0    1.0
            1    2.0
            2    3.0
            dtype: float64

        Returns:
            bigframes.series.Series:
                Return the y location (latitude) of point geometries.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
