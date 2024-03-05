from typing import Sequence

from bigframes import constants


class PlotAccessor:
    def hist(self, by: Sequence[str] | None = None, bins: int = 10, **kwargs):
        """
        Draw histogram of the input series using matplotlib.

        Parameters
        ----------
        by : str or sequence, optional
            If passed, then used to form histograms for separate groups.
            Currently, it is not supported yet.
        bins : int, default 10
            Number of histogram bins to be used.
        ax : matplotlib axes object, default None
            An axes of the current figure.
        grid : bool, default None (matlab style default)
            Axis grid lines.
        xticks : sequence
            Values to use for the xticks.
        yticks : sequence
            Values to use for the yticks.
        figsize : a tuple (width, height) in inches
            Size of a figure object.
        backend : str, default None
            Backend to use instead of the backend specified in the option
            ``plotting.backend``. Currently, only `matplotlib` is not supported yet.
        legend : bool, default False
            Place legend on axis subplots.
        **kwargs
            Options to pass to matplotlib plotting method.

        Returns
        -------
        class:`matplotlib.Axes`
            A histogram plot.

        Examples
        --------
        For Series:

        .. plot::
            :context: close-figs

            >>> import bigframes.pandas as bpd
            >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
            >>> ser = bpd.Series([1, 2, 2, 4, 6, 6], index=lst)
            >>> hist = ser.plot.hist()
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
