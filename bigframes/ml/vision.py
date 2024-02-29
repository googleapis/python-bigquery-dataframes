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

"""LLM models."""

from __future__ import annotations

from typing import Optional, Union

import vertexai
from vertexai import generative_models

import bigframes
from bigframes import clients, constants
from bigframes.ml import core, globals, utils
import bigframes.pandas as bpd


class MobileNetV2ImageAnnotator:
    def __init__(
        self,
        *,
        session: Optional[bigframes.Session] = None,
        connection_name: Optional[str] = None,
    ):
        self.session = session or bpd.get_global_session()
        self._bq_connection_manager = clients.BqConnectionManager(
            self.session.bqconnectionclient, self.session.resourcemanagerclient
        )

        connection_name = connection_name or self.session._bq_connection
        self.connection_name = self._bq_connection_manager.resolve_full_connection_name(
            connection_name,
            default_project=self.session._project,
            default_location=self.session._location,
        )

        self._bqml_model_factory = globals.bqml_model_factory()
        self._bqml_model: core.BqmlModel = self._create_bqml_model()

    def _create_bqml_model(self):
        # Parse and create connection if needed.
        if not self.connection_name:
            raise ValueError(
                "Must provide connection_name, either in constructor or through session options."
            )
        connection_name_parts = self.connection_name.split(".")
        if len(connection_name_parts) != 3:
            raise ValueError(
                f"connection_name must be of the format <PROJECT_NUMBER/PROJECT_ID>.<LOCATION>.<CONNECTION_ID>, got {self.connection_name}."
            )
        self._bq_connection_manager.create_bq_connection(
            project_id=connection_name_parts[0],
            location=connection_name_parts[1],
            connection_id=connection_name_parts[2],
            iam_role="serviceusage.serviceUsageConsumer",
        )

        options = {
            "REMOTE_SERVICE_TYPE": "CLOUD_AI_VISION_V1",
        }

        return self._bqml_model_factory.create_remote_model(
            session=self.session, connection_name=self.connection_name, options=options
        )

    def predict(
        self,
        X: Union[bpd.DataFrame, bpd.Series],
    ) -> bpd.DataFrame:
        (X,) = utils.convert_to_dataframe(X)

        if len(X.columns) != 1:
            raise ValueError(
                f"Only support one column as input. {constants.FEEDBACK_LINK}"
            )

        options = {"vision_features": ["label_detection"]}
        df = self._bqml_model.annotate_image(X, options)

        return df


class GeminiMultimodalTextGenerator:
    def __init__(
        self,
        *,
        session: Optional[bigframes.Session] = None,
    ):
        self.session = session or bpd.get_global_session()

    def predict(self, X: Union[bpd.DataFrame, bpd.Series]) -> bpd.DataFrame:
        (X,) = utils.convert_to_dataframe(X)

        vertexai.init(project=self.session._project, location="us-central1")
        multimodal_model = generative_models.GenerativeModel("gemini-1.0-pro-vision")

        responses = []
        for _, row in X.iterrows():
            uri = row.iloc[0]
            text = row.iloc[1]
            response = multimodal_model.generate_content(
                [
                    generative_models.Part.from_uri(uri, mime_type="image/jpeg"),
                    # Add an example query
                    text,
                ]
            )
            responses.append(response.text)

        X["multimodal_result"] = responses  # type: ignore

        return X
