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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bigframes

import bigframes.ml.api_primitives
import bigframes.ml.core

_REMOTE_LLM_MODEL_CODE = "CLOUD_AI_LARGE_LANGUAGE_MODEL_V1"


class PaLMTextGenerator(bigframes.ml.api_primitives.BaseEstimator):
    """PaLM text generator LLM model.

    Args:
        connection_name: connection to connect with remote service. str of the format <PROJECT_NUMBER/PROJECT_ID>.<REGION>.<CONNECTION_NAME>"""

    def __init__(self, connection_name: str):
        self.connection_name = connection_name
        self._bqml_model: bigframes.ml.core.BqmlModel = self._create_bqml_model()

    def _create_bqml_model(self):
        options = {"remote_service_type": _REMOTE_LLM_MODEL_CODE}

        return bigframes.ml.core.create_bqml_remote_model(
            connection_name=self.connection_name, options=options
        )
