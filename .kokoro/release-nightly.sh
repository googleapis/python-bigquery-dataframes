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
rm -rf build dist

# Workaround the fact that the repository that has been fetched before the
# build script. See: go/kokoro-native-docker-migration#known-issues and
# internal issue b/261050975.
git config --global --add safe.directory "${PROJECT_ROOT}"

python3 -m pip install --require-hashes -r .kokoro/requirements.txt

# Disable buffering, so that the logs stream through.
export PYTHONUNBUFFERED=1

# Install dependencies, as the following steps depend on it
pip install -e .[all]

# If NOX_SESSION is set, it only runs the specified session,
# otherwise run all the sessions.
if [[ -n "${NOX_SESSION:-}" ]]; then
    python3 -m nox -s ${NOX_SESSION:-}
else
    python3 -m nox
fi

# Update version string to include git hash and date
CURRENT_DATE=$(date '+%Y%m%d')
GIT_HASH=$(git rev-parse --short HEAD)
BIGFRAMES_VERSION=$(python3 -c "import bigframes; print(bigframes.__version__)")
RELEASE_VERSION=${BIGFRAMES_VERSION}dev${CURRENT_DATE}+${GIT_HASH}
sed -i -e "s/$BIGFRAMES_VERSION/$RELEASE_VERSION/g" bigframes/version.py

python3 setup.py sdist bdist_wheel

# Undo the version string edit, in case this script is running on a
# non-temporary instance of the bigframes repo
sed -i -e "s/$RELEASE_VERSION/$BIGFRAMES_VERSION/g" bigframes/version.py

LATEST_WHEEL=dist/bigframes-latest-py2.py3-none-any.whl
cp dist/bigframes-*.whl $LATEST_WHEEL
cp dist/bigframes-*.tar.gz dist/bigframes-latest.tar.gz

# Move into the package, build the distribution and upload to shared bucket.
# See internal bug 274624240 for details.

for gcs_path in gs://vertex_sdk_private_releases/bigframe/ \
                gs://dl-platform-colab/bigframes/;
do
  gsutil cp dist/* $gcs_path
  gsutil cp "notebooks/00 - Summary.ipynb" $gcs_path/notebooks/
done

# publish API coverage information to BigQuery
# Note: only the kokoro service account has permission to write to this
# table, if you want to test this step, point it to a table you have
# write access to
COVERAGE_TABLE=bigframes-metrics.coverage_report.bigframes_coverage_nightly
python3 publish_api_coverage.py \
  --bigframes_version=$BIGFRAMES_VERSION \
  --release_version=$RELEASE_VERSION \
  --bigquery_table=$COVERAGE_TABLE
