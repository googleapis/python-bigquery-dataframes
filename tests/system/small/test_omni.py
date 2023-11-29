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

import bigframes.pandas as pd


def test_cross_cloud_join():
    pd.options.bigquery.project = "cloud-endor-dev"
    pd.options.bigquery.location = "us"

    aws = (
        pd.read_gbq("cloud-endor-dev.demo_xcloud_aws.lineitem", index_col="l_orderkey")
        .groupby("l_orderkey")
        .count()
    )._cached()  # cached
    gdp = pd.read_gbq("cloud-endor-dev.demo_xcloud_us.orders", index_col="o_orderkey")

    aws.join(gdp).head(30).to_pandas()


def test_omni_aggregate_to_local():
    pd.options.bigquery.project = "cloud-endor-dev"
    pd.options.bigquery.location = "us"

    (
        pd.read_gbq("cloud-endor-dev.demo_xcloud_aws.lineitem", index_col="l_orderkey")
        .groupby("l_orderkey")
        .count()
        .head(30)
        .to_pandas(ordered=False)
    )


def test_omni_aggregate_to_gbq():
    pd.options.bigquery.project = "cloud-endor-dev"
    pd.options.bigquery.location = "us"

    (
        pd.read_gbq("cloud-endor-dev.demo_xcloud_aws.lineitem", index_col="l_orderkey")
        .groupby("l_orderkey")
        .count()
        .head(30)
        ._cached()
        .to_gbq()
    )
