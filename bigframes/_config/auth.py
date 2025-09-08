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

from __future__ import annotations

from typing import Optional

import google.auth.credentials
import pydata_google_auth

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def get_default_credentials_with_project() -> tuple[
    google.auth.credentials.Credentials, Optional[str]
]:
    return pydata_google_auth.default(scopes=_SCOPES, use_local_webserver=False)
