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

import bigframes
import bigframes.pandas as bpd


def test_blob_read_url(images_mm_df: bpd.DataFrame):
    bigframes.options.experiments.blob = True

    urls = images_mm_df["blob_col"].blob.read_url()

    assert urls.str.startswith("https://storage.googleapis.com/").all()


def test_blob_write_url(images_mm_df: bpd.DataFrame):
    bigframes.options.experiments.blob = True

    urls = images_mm_df["blob_col"].blob.write_url()

    assert urls.str.startswith("https://storage.googleapis.com/").all()
