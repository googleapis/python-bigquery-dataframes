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
from typing import Sequence

import matplotlib.pyplot as plt

import bigframes.constants as constants
import third_party.bigframes_vendored.pandas.plotting._core as vendordt


class PlotAccessor:
    __doc__ = vendordt.PlotAccessor.__doc__

    def __init__(self, data) -> None:
        self._parent = data

    def hist(self, by: Sequence[str] | None = None, bins: int = 10, **kwargs):
        if by is not None:
            raise NotImplementedError(
                f"Non-none `by` argument is not yet supported. {constants.FEEDBACK_LINK}"
            )
        if kwargs.pop("backend", None) is not None:
            raise NotImplementedError(
                f"Only support matplotlib backend for now. {constants.FEEDBACK_LINK}"
            )
        import bigframes.dataframe as dataframe

        if isinstance(self._parent, dataframe.DataFrame):
            raise NotImplementedError(
                f"`Dataframe.plot.hist` is not implemented yet. {constants.FEEDBACK_LINK}"
            )

        return self._hist_series(
            by=by,
            bins=bins,
            **kwargs,
        )

    def _hist_series(
        self,
        by: Sequence[str] | None = None,
        bins: int = 10,
        **kwargs,
    ):
        # Only supported some arguments to adorn plots.
        ax = kwargs.pop("ax", None)
        figsize = kwargs.pop("figsize", None)
        legend = kwargs.pop("legend", False)
        grid = kwargs.pop("grid", None)
        xticks = kwargs.pop("xticks", None)
        yticks = kwargs.pop("yticks", None)

        # Calculates the bins' values and weights through BigQuery
        import bigframes.pandas as bpd

        series = self._parent.copy()
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
        bin_edges = left_bins.union(right_bins, sort=True)

        # This code takes the hist_series function from pandas and tweaks it a bit.
        if kwargs.get("layout", None) is not None:
            raise ValueError("The 'layout' keyword is not supported when 'by' is None")

        fig = kwargs.pop(
            "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
        )
        if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
            fig.set_size_inches(*figsize, forward=True)

        ax = kwargs.pop("ax", None)
        if ax is None:
            ax = fig.gca()
        elif ax.get_figure() != fig:
            raise AssertionError("passed axis not bound to passed figure")

        if legend:
            kwargs["label"] = series.name
        ax.hist(x=left_bins, bins=bin_edges, weights=weights, **kwargs)
        if legend:
            ax.legend()
        if grid is not None:
            ax.grid(grid)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

        return ax
