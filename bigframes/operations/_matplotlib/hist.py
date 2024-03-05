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

from matplotlib.axes import Axes
import numpy as np

from third_party.bigframes_vendored.pandas.plotting._matplotlib.core import (
    LinePlot,
    MPLPlot,
)


# Based on pandas.plot.Histplot (see link) with modifications:
# https://github.com/pandas-dev/pandas/blob/v2.1.3/pandas/plotting/_matplotlib/hist.py
class BFHistPlot(LinePlot):
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

        # TODO(chelsealin): Support timestamp/date types here.
        exclude_type = None
        include_type = ["number"]
        numeric_data = data.select_dtypes(include=include_type, exclude=exclude_type)
        try:
            is_empty = numeric_data.columns.empty
        except AttributeError:
            is_empty = not len(numeric_data)

        # no non-numeric frames or series allowed
        if is_empty:
            raise TypeError("no numeric data to plot")

        self.data = numeric_data
        self.binned_data = self._calculate_binned_data(self.data)

        # calculate bin number separately in different subplots
        # where subplots are created based on by argument
        self._calculate_subplots_bins(self.data)

    def _calculate_binned_data(self, data):
        import bigframes.pandas as bpd

        # TODO: Optimize it by a batching job.
        binned_data = []
        for _, col in enumerate(data.columns):
            cutted_data = bpd.cut(data[col], bins=self.bins, labels=None)
            binned_data.append(
                cutted_data.struct.explode()
                .value_counts()
                .to_pandas()
                .sort_index(level="left_exclusive")
            )
        return binned_data

    def _calculate_subplots_bins(self, data) -> np.ndarray:
        """Calculate bins given data"""
        if self.subplots:
            if isinstance(self.subplots, bool):
                self.subplots_bins = list(range(len(data.columns)))
                for col_idx in range(len(data.columns)):
                    bins = (
                        self.binned_data[col_idx]
                        .index.get_level_values("left_exclusive")
                        .union(
                            self.binned_data[col_idx].index.get_level_values(
                                "right_inclusive"
                            )
                        )
                    )
                    _, self.subplots_bins[col_idx] = np.histogram(
                        bins, bins=self.bins, range=self.kwds.get("range", None)
                    )
            elif isinstance(self.subplots, list):
                self.subplots_bins = list(range(len(self.subplots)))
                for (group_idx, group) in enumerate(self.subplots):
                    bins = []
                    for col_idx in group:
                        bins = (
                            self.binned_data[col_idx]
                            .index.get_level_values("left_exclusive")
                            .union(bins)
                        )
                        bins = (
                            self.binned_data[col_idx]
                            .index.get_level_values("right_inclusive")
                            .union(bins)
                        )
                    _, self.subplots_bins[group_idx] = np.histogram(
                        bins, bins=self.bins, range=self.kwds.get("range", None)
                    )
        else:
            bins = []
            for col_idx, _ in enumerate(data.columns):
                bins = (
                    self.binned_data[col_idx]
                    .index.get_level_values("left_exclusive")
                    .union(bins)
                )
                bins = (
                    self.binned_data[col_idx]
                    .index.get_level_values("right_inclusive")
                    .union(bins)
                )
            _, self.subplots_bins = np.histogram(
                bins, bins=self.bins, range=self.kwds.get("range", None)
            )

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
                column_num=i,
                stacking_id=stacking_id,
                **kwds,
                x=self.binned_data[i].index.get_level_values("left_exclusive"),
                bins=self._get_ax_bins(i),
                weights=self.binned_data[i].values,
            )
            self._append_legend_handles_labels(artists[0], label)

    def _get_ax_bins(self, i: int):
        if self.subplots:
            i = self._col_idx_to_axis_idx(i)
            return self.subplots_bins[i]
        else:
            return self.subplots_bins

    # error: Signature of "_plot" incompatible with supertype "LinePlot"
    @classmethod
    def _plot(  # type: ignore[override]
        self,
        ax,
        column_num: int = 0,
        stacking_id=None,
        *,
        x,
        bins,
        weights,
        **kwds,
    ):
        if column_num == 0:
            self._initialize_stacker(ax, stacking_id, len(bins) - 1)

        print(f"bins: {bins}")
        print(f"weights: {weights}")
        print(f"x: {x}")

        n, bins, patches = ax.hist(x=x, bins=bins, weights=weights, **kwds)

        self._update_stacker(ax, stacking_id, n)
        return patches

    def _post_plot_logic(self, ax: Axes, data) -> None:
        if self.orientation == "horizontal":
            ax.set_xlabel("Frequency" if self.xlabel is None else self.xlabel)
            ax.set_ylabel(self.ylabel)
        else:
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel("Frequency" if self.ylabel is None else self.ylabel)

    @property
    def orientation(self) -> str:
        if self.kwds.get("orientation", None) == "horizontal":
            return "horizontal"
        else:
            return "vertical"
