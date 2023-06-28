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

import google.api_core.exceptions
import pytest


@pytest.mark.parametrize("missing_parts_table_id", [(""), ("table")])
def test_read_gbq_missing_parts(session, missing_parts_table_id):
    with pytest.raises(ValueError):
        session.read_gbq(missing_parts_table_id)


@pytest.mark.parametrize(
    "not_found_table_id",
    [("unknown.dataset.table"), ("project.unknown.table"), ("project.dataset.unknown")],
)
def test_read_gdb_not_found_tables(session, not_found_table_id):
    with pytest.raises(google.api_core.exceptions.NotFound):
        session.read_gbq(not_found_table_id)
