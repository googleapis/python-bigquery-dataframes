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

import itertools
from typing import Literal

import numpy as np
import pandas as pd

import bigframes.constants as constants
from bigframes.operations._matplotlib.core import MPLPlot


class HistPlot(MPLPlot):
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
        self.label = kwargs.get("label", None)
        self.by = kwargs.pop("by", None)
        self.kwargs = kwargs

        if self.by is not None:
            raise NotImplementedError(
                f"Non-none `by` argument is not yet supported. {constants.FEEDBACK_LINK}"
            )
        if not isinstance(self.bins, int):
            raise NotImplementedError(
                f"Only integer values are supported for the `bins` argument. {constants.FEEDBACK_LINK}"
            )
        if kwargs.get("weight", None) is not None:
            raise NotImplementedError(
                f"Non-none `weight` argument is not yet supported. {constants.FEEDBACK_LINK}"
            )

        self.data = self._compute_plot_data(data)

    def generate(self) -> None:
        hist_bars = self._calculate_hist_bar(self.data, self.bins)

        bin_edges = None
        hist_x = {}
        weights = {}
        for col_name, hist_bar in hist_bars.items():
            left = hist_bar.index.get_level_values("left_exclusive")
            right = hist_bar.index.get_level_values("right_inclusive")

            hist_x[col_name] = pd.Series((left + right) / 2.0)
            weights[col_name] = hist_bar.values
            if bin_edges is None:
                bin_edges = left.union(right)
            else:
                bin_edges = left.union(right).union(bin_edges)

        bins = None
        if bin_edges is not None:
            _, bins = np.histogram(
                bin_edges, bins=self.bins, range=self.kwargs.get("range", None)
            )

        # Fills with NA values when items have different lengths.
        ordered_columns = self.data.columns.values
        hist_x_pd = pd.DataFrame(
            list(itertools.zip_longest(*hist_x.values())), columns=list(hist_x.keys())
        ).sort_index(axis=1)
        weights_pd = pd.DataFrame(
            list(itertools.zip_longest(*weights.values())), columns=list(weights.keys())
        ).sort_index(axis=1)

        self.axes = hist_x_pd[ordered_columns].plot.hist(
            bins=bins,
            weights=np.array(weights_pd[ordered_columns].values),
            **self.kwargs,
        )  # type: ignore

    def _compute_plot_data(self, data):
        # Importing at the top of the file causes a circular import.
        import bigframes.series as series

        if isinstance(data, series.Series):
            label = self.label
            if label is None and data.name is None:
                label = ""
            if label is None:
                data = data.to_frame()
            else:
                data = data.to_frame(name=label)

        # TODO(chelsealin): Support timestamp/date types here.
        include_type = ["number"]
        numeric_data = data.select_dtypes(include=include_type)
        try:
            is_empty = numeric_data.columns.empty
        except AttributeError:
            is_empty = not len(numeric_data)

        if is_empty:
            raise TypeError("no numeric data to plot")

        return numeric_data

    @staticmethod
    def _calculate_hist_bar(data, bins):
        import bigframes.pandas as bpd

        # TODO: Optimize this by batching multiple jobs into one.
        hist_bar = {}
        for _, col in enumerate(data.columns):
            cutted_data = bpd.cut(data[col], bins=bins, labels=None)
            hist_bar[col] = (
                cutted_data.struct.explode()
                .value_counts()
                .to_pandas()
                .sort_index(level="left_exclusive")
            )
        return hist_bar
