# Copyright 2024 Google LLC
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

import bigframes.constants

ENGINE_ERROR_TEMPLATE = (
    "write_engine='{write_engine}' is incompatible with engine='{engine}'. "
    + bigframes.constants.FEEDBACK_LINK
)


def validate_engine_compatibility(engine, write_engine):
    """Raises NotImplementedError if engine is not compatible with write_engine."""

    if engine == "bigquery" and write_engine in (
        "bigquery_inline",
        "bigquery_streaming",
    ):
        raise NotImplementedError(
            ENGINE_ERROR_TEMPLATE.format(engine=engine, write_engine=write_engine)
        )

    if engine != "bigquery" and write_engine in ("bigquery_external_table",):
        raise NotImplementedError(
            ENGINE_ERROR_TEMPLATE.format(engine=engine, write_engine=write_engine)
        )
