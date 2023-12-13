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

"""Options for downsampling."""

import dataclasses

import third_party.bigframes_vendored.pandas.core.config_init as vendored_pandas_config


@dataclasses.dataclass
class OrderingOptions:
    __doc__ = vendored_pandas_config.ordering_options_doc

    enabled: bool = True
