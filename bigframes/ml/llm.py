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

from typing import cast, TYPE_CHECKING

if TYPE_CHECKING:
    import bigframes

import bigframes.ml.api_primitives
import bigframes.ml.core

_REMOTE_LLM_MODEL_CODE = "CLOUD_AI_LARGE_LANGUAGE_MODEL_V1"
_VERTEX_ENDPOINT_FORMAT_US_CENTRAL1 = "https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/text-bison"
_TEXT_GENERATE_RESULT_COLUMN = "ml_generate_text_result"


class PaLM2TextGenerator(bigframes.ml.api_primitives.BaseEstimator):
    """PaLM2 text generator LLM model.

    Args:
        session: BQ session to create the model
        connection_name: connection to connect with remote service. str of the format <PROJECT_NUMBER/PROJECT_ID>.<REGION>.<CONNECTION_NAME>"""

    def __init__(self, session: bigframes.Session, connection_name: str):
        self.session = session
        self.connection_name = connection_name
        self._bqml_model: bigframes.ml.core.BqmlModel = self._create_bqml_model()

    def _create_bqml_model(self):
        project_id = self.session.bqclient.project
        options = {
            "remote_service_type": _REMOTE_LLM_MODEL_CODE,
            "endpoint": _VERTEX_ENDPOINT_FORMAT_US_CENTRAL1.format(
                project_id=project_id
            ),
        }

        return bigframes.ml.core.create_bqml_remote_model(
            session=self.session, connection_name=self.connection_name, options=options
        )

    def predict(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        """Predict the result from input DataFrame.

        Args:
            X: input DataFrame, which needs to contain a column with name "prompt". Only the column will be used as input.

        Returns: output DataFrame with only 1 column as the JSON output results."""
        df = self._bqml_model.generate_text(X)
        return cast(
            bigframes.DataFrame,
            df[[_TEXT_GENERATE_RESULT_COLUMN]],
        )
