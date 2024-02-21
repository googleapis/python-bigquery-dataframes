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

# import bigframes.pandas
import matplotlib.pyplot as plt


class PlottingHelper:
    def __init__(self, frame=None):
        self._frame = frame


def hist_series(self, bins=None, **kwargs):
    import bigframes.pandas

    binned = bigframes.pandas.cut(self, bins=bins, labels=None)
    # convert struct to dataframe and count unique values
    binned_data = (
        binned.struct.explode()
        .value_counts()
        .to_pandas()
        .sort_index(level="left_exclusive")
    )

    # Combine all bin edges
    left_bins = binned_data.index.get_level_values("left_exclusive")
    right_bins = binned_data.index.get_level_values("right_inclusive")
    bin_edges = left_bins.union(right_bins, sort=True)

    # obtain weights
    weights = binned_data.values

    return plt.hist(x=left_bins, bins=bin_edges, weights=weights)


# def hist_dataframe(self: bigframes.dataframe.DataFrame, bins=None, **kwargs):
#    return "Not implemented"
