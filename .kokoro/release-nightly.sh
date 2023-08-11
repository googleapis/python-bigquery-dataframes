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
set -x

# Parse command line arguments
DRY_RUN=
while [ $# -gt 0 ] ; do
  case "$1" in
    -d | --dry-run )
      DRY_RUN=true
      ;;
    -h | --help )
      echo -e "USAGE: `basename $0` [ -d | --dry-run ]"
      exit
      ;;
  esac
  shift 1;
done

if [ -z "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="${KOKORO_ARTIFACTS_DIR}/git/bigframes"
fi

# Move into the package, build the distribution and upload to shared bucket.
# See internal bug 274624240 for details.

cd "${PROJECT_ROOT}"
rm -rf build dist

# Workaround the fact that the repository that has been fetched before the
# build script. See: go/kokoro-native-docker-migration#known-issues and
# internal issue b/261050975.
git config --global --add safe.directory "${PROJECT_ROOT}"

python3.10 -m pip install --require-hashes -r .kokoro/requirements.txt

# Disable buffering, so that the logs stream through.
export PYTHONUNBUFFERED=1

# Install dependencies, as the following steps depend on it
python3.10 -m pip install -e .[all]

# If NOX_SESSION is set, it only runs the specified session,
# otherwise run all the sessions.
if ! [ ${DRY_RUN} ]; then
    if [ -n "${NOX_SESSION:-}" ]; then
        python3.10 -m nox -s ${NOX_SESSION:-}
    else
        python3.10 -m nox
    fi
fi

# Generate third party notices and include it in the licenses in setup.cfg
# TODO(shobs): Don't include it in the package once vertex colab can pick it
# from elsewhere
THIRD_PARTY_NOTICES_FILE=THIRD_PARTY_NOTICES
python3.10 -m pip install pip-licenses
python3.10 scripts/generate_third_party_notices.py --output-file ${THIRD_PARTY_NOTICES_FILE}
if ! [ -s ${THIRD_PARTY_NOTICES_FILE} ]; then
    echo "${THIRD_PARTY_NOTICES_FILE} was generated with zero size"
    exit -1
fi
SETUP_CFG_BKP=`mktemp`
cp -f setup.cfg ${SETUP_CFG_BKP}
cat >> setup.cfg << EOF

[metadata]
license_files =
    LICENSE
    ${THIRD_PARTY_NOTICES_FILE}
EOF

# Update version string to include git hash and date
CURRENT_DATE=$(date '+%Y%m%d')
GIT_HASH=$(git rev-parse --short HEAD)
BIGFRAMES_VERSION=$(python3.10 -c "import bigframes; print(bigframes.__version__)")
RELEASE_VERSION=${BIGFRAMES_VERSION}dev${CURRENT_DATE}+${GIT_HASH}
sed -i -e "s/$BIGFRAMES_VERSION/$RELEASE_VERSION/g" bigframes/version.py

# Generate the package wheel
python3.10 setup.py sdist bdist_wheel

# Make sure that the wheel file is generated
VERSION_WHEEL=`ls dist/bigframes-*.whl`
num_wheel_files=`echo $VERSION_WHEEL | wc -w`
if [ $num_wheel_files -ne 1 ] ; then
    echo "Exactly one wheel file should have been generated, found $num_wheel_files: $VERSION_WHEEL"
    exit -1
fi

# Make sure the wheel file has the third party notices included
# TODO(shobs): An utimate validation would be to create a virtual environment
# and install the wheel file, then verify that
# site-packages/bigframes-*.dist-info/ includes third party notices
python3.10 -c "
from zipfile import ZipFile
with ZipFile('$VERSION_WHEEL') as myzip:
    third_party_licenses_info = [
        info
        for info in myzip.infolist()
        if info.filename.endswith('.dist-info/${THIRD_PARTY_NOTICES_FILE}')
    ]
    assert (
        len(third_party_licenses_info) == 1
    ), f'Found {len(third_party_licenses_info)} third party licenses'
    assert (
        third_party_licenses_info[0].file_size > 0
    ), 'Package contains third party license of size 0'
"

# Create a copy of the wheel with a well known, version agnostic name
LATEST_WHEEL=dist/bigframes-latest-py2.py3-none-any.whl
cp $VERSION_WHEEL $LATEST_WHEEL
cp dist/bigframes-*.tar.gz dist/bigframes-latest.tar.gz

if ! [ ${DRY_RUN} ]; then
    for gcs_path in gs://vertex_sdk_private_releases/bigframe/ \
                    gs://dl-platform-colab/bigframes/ \
                    gs://bigframes-wheels/;
    do
      gsutil cp -v dist/* ${gcs_path}
      gsutil cp -v LICENSE ${gcs_path}
      gsutil cp -v ${THIRD_PARTY_NOTICES_FILE} ${gcs_path}
      gsutil -m cp -v "notebooks/00 - Summary.ipynb" \
                      "notebooks/01 - Getting Started.ipynb" \
                      "notebooks/02 - DataFrame.ipynb" \
                      "notebooks/03 - Using ML - ML fundamentals.ipynb" \
                      "notebooks/04 - Using ML - SKLearn linear regression.ipynb" \
                      "notebooks/05 - Using ML - Easy linear regression.ipynb" \
                      "notebooks/06 - Using ML - Large Language Models.ipynb" \
                      "notebooks/50 - Remote Function.ipynb" \
                      ${gcs_path}notebooks/
    done

    # publish API coverage information to BigQuery
    # Note: only the kokoro service account has permission to write to this
    # table, if you want to test this step, point it to a table you have
    # write access to
    COVERAGE_TABLE=bigframes-metrics.coverage_report.bigframes_coverage_nightly
    python3.10 scripts/publish_api_coverage.py \
      --bigframes_version=$BIGFRAMES_VERSION \
      --release_version=$RELEASE_VERSION \
      --bigquery_table=$COVERAGE_TABLE
fi

# Undo the file changes, in case this script is running on a
# non-temporary instance of the bigframes repo
# TODO: This doesn't work with (set -eo pipefail) if the failure happened after
# the changes were made but before this cleanup, because the script would
# terminate with the failure itself. See if we can ensure the cleanup.
sed -i -e "s/$RELEASE_VERSION/$BIGFRAMES_VERSION/g" bigframes/version.py
mv -f ${SETUP_CFG_BKP} setup.cfg
rm -f ${THIRD_PARTY_NOTICES_FILE}

# Keep this last so as not to block the release on PDF docs build.
pdf_docs () {
    sudo apt update
    sudo apt install -y texlive texlive-latex-extra latexmk

    pushd "${PROJECT_ROOT}/docs"
    make latexpdf

    cp "_build/latex/bigframes.pdf" "_build/latex/bigframes-${RELEASE_VERSION}.pdf"
    cp "_build/latex/bigframes.pdf" "_build/latex/bigframes-latest.pdf"

    if ! [ ${DRY_RUN} ]; then
        for gcs_path in gs://vertex_sdk_private_releases/bigframe/ \
                        gs://dl-platform-colab/bigframes/ \
                        gs://bigframes-wheels/;
        do
          gsutil cp -v "_build/latex/bigframes-*.pdf" ${gcs_path}
        done
    fi

    popd
}

pdf_docs

# Copy html docs to GCS from where it can be deployed to anywhere else
gcs_docs () {
    docs_gcs_bucket=gs://bigframes-docs
    docs_local_html_folder=docs/_build/html
    if [ ! -d ${docs_local_html_folder} ]; then
      python3.10 -m nox -s docs
    fi

    if ! [ ${DRY_RUN} ]; then
        gsutil -m cp -v -r ${docs_local_html_folder} ${docs_gcs_bucket}/${GIT_HASH}

        # Copy the script to refresh firebase docs website from GCS to GCS itself
        gsutil -m cp -v scripts/update_firebase_docs_site.sh ${docs_gcs_bucket}
    fi
}

gcs_docs

if ! [ ${DRY_RUN} ]; then
    # Copy docs and wheels to Google Drive
    python3.10 scripts/upload_to_google_drive.py
fi
