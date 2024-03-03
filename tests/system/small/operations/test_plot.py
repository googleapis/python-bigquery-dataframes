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

import pandas._testing as tm


def test_series_hist_bins(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    ax = scalars_df["int64_col"].plot.hist(bins=5)
    pd_ax = scalars_pandas_df["int64_col"].hist(bins=5)

    # Check hist has same height compared to the pandas one.
    assert len(ax.patches) == len(pd_ax.patches)
    for i in range(len(ax.patches)):
        assert ax.patches[i].xy == pd_ax.patches[i].xy
        assert ax.patches[i]._height == pd_ax.patches[i]._height


def test_series_hist_ticks_props(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    xticks = [20, 18]
    yticks = [30, 40]

    ax = scalars_df["float64_col"].plot.hist(xticks=xticks, yticks=yticks)
    pd_ax = scalars_pandas_df["float64_col"].plot.hist(xticks=xticks, yticks=yticks)
    xlabels = ax.get_xticklabels()
    pd_xlables = pd_ax.get_xticklabels()
    assert len(xlabels) == len(pd_xlables)
    for i in range(len(pd_xlables)):
        tm.assert_almost_equal(xlabels[i].get_fontsize(), pd_xlables[i].get_fontsize())
        tm.assert_almost_equal(xlabels[i].get_rotation(), pd_xlables[i].get_rotation())

    ylabels = ax.get_yticklabels()
    pd_ylables = pd_ax.get_yticklabels()
    assert len(xlabels) == len(pd_xlables)
    for i in range(len(pd_xlables)):
        tm.assert_almost_equal(ylabels[i].get_fontsize(), pd_ylables[i].get_fontsize())
        tm.assert_almost_equal(ylabels[i].get_rotation(), pd_ylables[i].get_rotation())
