# Copyright 2025 Google LLC
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

import datetime
import importlib

import pytest

from bigframes import version


def test_warning_when_release_date_is_too_old(monkeypatch):
    release_date = datetime.datetime.strptime(version.__release_date__, "%Y-%m-%d")
    current_date = release_date + datetime.timedelta(days=366)

    class FakeDatetime(datetime.datetime):
        @classmethod
        def today(cls):
            return current_date

    monkeypatch.setattr(datetime, "datetime", FakeDatetime)

    with pytest.warns(Warning, match=r".+ Please update to the lastest version"):
        importlib.reload(version)
