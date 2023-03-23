#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Based loosely on
# https://github.com/googleapis/python-bigquery/blob/main/.kokoro/release.sh

set -eo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="${KOKORO_ARTIFACTS_DIR}/git/bigframes"
fi

cd "${PROJECT_ROOT}"

python3 -m pip install --require-hashes -r .kokoro/requirements.txt

# Disable buffering, so that the logs stream through.
export PYTHONUNBUFFERED=1

# Update version string to include git hash and date
CURRENT_DATE=$(date '+%Y%m%d')
GIT_HASH=$(git rev-parse --short HEAD)
sed -i -E \
  "s/__version__ = \"([0-9]+\.[0-9]+\.[0-9]+)[^0-9]*\"/__version__ = \"\1dev${CURRENT_DATE}+${GIT_HASH}\"/" \
  bigframes/version.py

cp dist/bigframes-*.whl dist/bigframes-latest.whl
cp dist/bigframes-*.tar.gz dist/bigframes-latest.tar.gz

# Move into the package, build the distribution and upload to shared bucket.
# See internal bug 274624240 for details.
python3 setup.py sdist bdist_wheel
gsutil cp dist/* gs://vertex_sdk_private_releases/bigframe/
