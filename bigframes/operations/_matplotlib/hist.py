# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal

from pandas.plotting._matplotlib.core import MPLPlot
from pandas.plotting._matplotlib.hist import HistPlot


class BFHistPlot(HistPlot):
    @property
    def _kind(self) -> Literal["hist"]:
        return "hist"

    def __init__(
        self,
        data,
        bins: int = 10,
        **kwargs,
    ) -> None:
        self.bins = bins
        self.xlabel = kwargs.get("xlabel")
        self.ylabel = kwargs.get("ylabel")
        # Do not call LinePlot.__init__ which may fill nan
        MPLPlot.__init__(self, data, **kwargs)  # pylint: disable=non-parent-init-called

    def _args_adjust(self) -> None:
        # TODO(chelsealin): calculate_bins for Dataframe type
        pass

    def _compute_plot_data(self):
        # Importing at the top of the file causes a circular import.
        import bigframes.series as series

        data = self.data
        if isinstance(data, series.Series):
            label = self.label
            if label is None and data.name is None:
                label = ""
            if label is None:
                data = data.to_frame()
            else:
                data = data.to_frame(name=label)

        # TODO(chelsealin): confirm the final include type.
        exclude_type = None
        include_type = ["number", "datetime", "datetimetz", "timedelta"]

        numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)
        try:
            is_empty = numeric_data.columns.empty
        except AttributeError:
            is_empty = not len(numeric_data)

        # no non-numeric frames or series allowed
        if is_empty:
            raise TypeError("no numeric data to plot")

        self.data = numeric_data.apply(self._convert_to_ndarray)

    def _make_plot(self) -> None:
        colors = self._get_colors()
        stacking_id = self._get_stacking_id()
        data = self.data

        for i, col in enumerate(data.columns):
            ax = self._get_ax(i)

            kwds = self.kwds.copy()

            label = self._mark_right_label(col, index=i)
            kwds["label"] = label

            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds["style"] = style

            artists = self._plot(
                ax,
                data[col],
                column_num=i,
                stacking_id=stacking_id,
                **kwds,
                bins=self.bins,
            )
            self._append_legend_handles_labels(artists[0], label)

    # error: Signature of "_plot" incompatible with supertype "LinePlot"
    @classmethod
    def _plot(  # type: ignore[override]
        self,
        ax,
        series,
        column_num: int = 0,
        stacking_id=None,
        *,
        bins,
        **kwds,
    ):
        if column_num == 0:
            self._initialize_stacker(ax, stacking_id, bins - 1)

        # TODO(chelsealin): Optimize it by batching jobs.
        x, hist_bins, weights = self._calculate_hist_bin(series, bins)
        n, bins, patches = ax.hist(x=x, bins=hist_bins, weights=weights, **kwds)

        self._update_stacker(ax, stacking_id, n)
        return patches

    @staticmethod
    def _calculate_hist_bin(series, bins: int):
        import bigframes.pandas as bpd

        binned = bpd.cut(series, bins=bins, labels=None)
        binned_data = (
            binned.struct.explode()
            .value_counts()
            .to_pandas()
            .sort_index(level="left_exclusive")
        )
        weights = binned_data.values
        left_bins = binned_data.index.get_level_values("left_exclusive")
        right_bins = binned_data.index.get_level_values("right_inclusive")
        hist_bins = left_bins.union(right_bins, sort=True)

        return right_bins, hist_bins, weights
