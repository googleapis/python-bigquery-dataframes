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

import abc
import typing
import uuid

import matplotlib.pyplot as plt
import pandas as pd

import bigframes.dtypes as dtypes

DEFAULT_SAMPLING_N = 1000
DEFAULT_SAMPLING_STATE = 0

class MPLPlot(abc.ABC):
    @abc.abstractmethod
    def generate(self):
        pass

    def draw(self) -> None:
        # This import can fail with "Matplotlib failed to acquire the
        # following lock file" so import here to reduce the chance of
        # our parallel test suite from triggering this.
        import matplotlib.pyplot as plt

        plt.draw_if_interactive()

    @property
    def result(self):
        return self.axes


class SamplingPlot(MPLPlot):
    @abc.abstractproperty
    def _kind(self):
        pass

    def __init__(self, data, **kwargs) -> None:
        self.kwargs = kwargs
        self.data = data

    def generate(self) -> None:
        plot_data = self._compute_plot_data()
        self.axes = plot_data.plot(kind=self._kind, **self.kwargs)

    def _compute_sample_data(self, data):
        # TODO: Cache the sampling data in the PlotAccessor.
        sampling_n = self.kwargs.pop("sampling_n", DEFAULT_SAMPLING_N)
        sampling_random_state = self.kwargs.pop(
            "sampling_random_state", DEFAULT_SAMPLING_STATE
        )
        return data.sample(
            n=sampling_n,
            random_state=sampling_random_state,
            sort=False,
        ).to_pandas()

    def _compute_plot_data(self):
        return self._compute_sample_data(self.data)


class LinePlot(SamplingPlot):
    @property
    def _kind(self) -> typing.Literal["line"]:
        return "line"


class AreaPlot(SamplingPlot):
    @property
    def _kind(self) -> typing.Literal["area"]:
        return "area"


class ScatterPlot(SamplingPlot):
    @property
    def _kind(self) -> typing.Literal["scatter"]:
        return "scatter"

    def __init__(self, data, **kwargs) -> None:
        super().__init__(data, **kwargs)

        c = self.kwargs.get("c", None)
        if self._is_sequence_arg(c) and len(c) != self.data.shape[0]:
            raise ValueError(
                f"'c' argument has {len(c)} elements, which is "
                + f"inconsistent with 'x' and 'y' with size {self.data.shape[0]}"
            )

    def _compute_plot_data(self):
        data = self.data.copy()

        c = self.kwargs.get("c", None)
        c_id = None
        if self._is_sequence_arg(c):
            c_id = self._generate_new_column_name(data)
            print(c_id)
            data[c_id] = c

        sample = self._compute_sample_data(data)

        # Works around a pandas bug:
        # https://github.com/pandas-dev/pandas/commit/45b937d64f6b7b6971856a47e379c7c87af7e00a
        if self._is_column_name(c, sample) and sample[c].dtype == dtypes.STRING_DTYPE:
            sample[c] = sample[c].astype("object")

        if c_id is not None:
            self.kwargs["c"] = sample[c_id]
            sample = sample.drop(columns=[c_id])

        return sample

    def _is_sequence_arg(self, arg):
        return (
            arg is not None
            and not isinstance(arg, str)
            and isinstance(arg, typing.Iterable)
        )

    def _is_column_name(self, arg, data):
        return (
            arg is not None
            and pd.core.dtypes.common.is_hashable(arg)
            and arg in data.columns
        )

    def _generate_new_column_name(self, data):
        col_name = None
        while col_name is None or col_name in data.columns:
            col_name = f"plot_temp_{str(uuid.uuid4())[:8]}"
        return col_name
