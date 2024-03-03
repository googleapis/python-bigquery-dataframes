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

import bigframes.constants as constants
import bigframes.operations._matplotlib as plotbackend
import third_party.bigframes_vendored.pandas.plotting._core as vendordt

# import matplotlib.pyplot as plt
# import pandas.plotting._matplotlib.tools as pd_plt


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
        return plotbackend.plot(self._parent.copy(), kind="hist", **kwargs)
        # import bigframes.dataframe as dataframe
        # import bigframes.series as series

        # if isinstance(self._parent, dataframe.DataFrame):
        #     return self._hist_frame(by=by, bins=bins, **kwargs)
        # elif isinstance(self._parent, series.Series):
        #     return self._hist_series(by=by, bins=bins, **kwargs)
        # else:
        #     raise TypeError(
        #         f"Unsupported type: {type(self).__name__}"
        #     )

    # def _hist_series(
    #     self,
    #     by: Sequence[str] | None = None,
    #     bins: int = 10,
    #     **kwargs,
    # ):
    #     # Only supported some arguments to adorn plots.
    #     ax = kwargs.pop("ax", None)
    #     figsize = kwargs.pop("figsize", None)
    #     legend = kwargs.pop("legend", False)
    #     grid = kwargs.pop("grid", None)
    #     xticks = kwargs.pop("xticks", None)
    #     yticks = kwargs.pop("yticks", None)

    #     # This code takes the hist_series function from pandas and tweaks it a bit.
    #     if kwargs.get("layout", None) is not None:
    #         raise ValueError("The 'layout' keyword is not supported when 'by' is None")

    #     fig = kwargs.pop(
    #         "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
    #     )
    #     if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
    #         fig.set_size_inches(*figsize, forward=True)

    #     ax = kwargs.pop("ax", None)
    #     if ax is None:
    #         ax = fig.gca()
    #     elif ax.get_figure() != fig:
    #         raise AssertionError("passed axis not bound to passed figure")

    #     series = self._parent.copy()
    #     if legend:
    #         kwargs["label"] = series.name

    #     x, hist_bins, weights = self._calculate_hist_bin(series, bins)
    #     ax.hist(x=x, bins=hist_bins, weights=weights, **kwargs)

    #     if legend:
    #         ax.legend()
    #     if grid is not None:
    #         ax.grid(grid)
    #     if xticks is not None:
    #         ax.set_xticks(xticks)
    #     if yticks is not None:
    #         ax.set_yticks(yticks)

    #     return ax

    # def _hist_frame(
    #     self,
    #     by: Sequence[str] | None = None,
    #     bins: int = 10,
    #     **kwargs,
    # ):
    #     # Only supported some arguments to adorn plots.
    #     ax = kwargs.pop("ax", None)
    #     figsize = kwargs.pop("figsize", None)
    #     legend = kwargs.pop("legend", False)
    #     grid = kwargs.pop("grid", None)
    #     xticks = kwargs.pop("xticks", None)
    #     yticks = kwargs.pop("yticks", None)

    #     # TODO: Make GH32590 works
    #     # TODO(chelsealin): check dataframes with index
    #     data = self._parent.copy()
    #     data = data.select_dtypes(
    #         include=("number", "datetime64", "datetimetz")
    #     )
    #     naxes = len(data.columns)
    #     if naxes == 0:
    #         raise ValueError(
    #             "hist method requires numerical or datetime columns, nothing to plot."
    #         )

    #     fig, axes = pd_plt.create_subplots(
    #         naxes=naxes,
    #         ax=ax,
    #         squeeze=False,
    #         sharex=False,
    #         sharey=False,
    #         figsize=figsize,
    #         layout=None,
    #     )
    #     _axes = pd_plt.flatten_axes(axes)

    #     can_set_label = "label" not in kwargs

    #     for i, col in enumerate(data.columns):
    #         ax = _axes[i]
    #         if legend and can_set_label:
    #             kwargs["label"] = col
    #         # TODO(chelsealin): check data[col].dropna() here!!!!
    #         x, hist_bins, weights = self._calculate_hist_bin(data[col], bins)
    #         ax.hist(x=x, bins=hist_bins, weights=weights, **kwargs)
    #         ax.set_title(col)
    #         ax.grid(grid)
    #         if legend:
    #             ax.legend()

    #     if xticks is not None:
    #         ax.set_xticks(xticks)
    #     if yticks is not None:
    #         ax.set_yticks(yticks)
    #     pd_plt.maybe_adjust_figure(fig, wspace=0.3, hspace=0.3)

    #     return axes

    # @staticmethod
    # def _calculate_hist_bin(series, bins: int):
    #     import bigframes.pandas as bpd
    #     binned = bpd.cut(series.copy(), bins=bins, labels=None)
    #     binned_data = (
    #         binned.struct.explode()
    #         .value_counts()
    #         .to_pandas()
    #         .sort_index(level="left_exclusive")
    #     )
    #     weights = binned_data.values
    #     left_bins = binned_data.index.get_level_values("left_exclusive")
    #     right_bins = binned_data.index.get_level_values("right_inclusive")
    #     hist_bins = left_bins.union(right_bins, sort=True)

    #     # TODO(chelsealin): check why it is left_bins rather than right_bins?
    #     return left_bins, hist_bins, weights
