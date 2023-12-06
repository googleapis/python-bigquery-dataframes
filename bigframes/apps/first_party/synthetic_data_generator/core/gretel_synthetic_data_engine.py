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

import warnings

import pandas as pd

import bigframes.pandas as bpd

try:
    import gretel_client
    from gretel_client import configure_session
    from gretel_client.helpers import poll
    from gretel_client.projects import create_or_get_unique_project
    from gretel_client.projects.models import read_model_config
except:
    gretel_client = None


class GretelSyntheticDataEngine:
    def __init__(self, api_key=None):
        if gretel_client is None:
            return
        if api_key is None:
            api_key = "prompt"
        configure_session(api_key=api_key, cache="yes", validate=True)
        self.project = create_or_get_unique_project(name="synthetic-data")

    def train_and_generate_synthetic_data(self, orig_df, num_rows=None):
        if gretel_client is None:
            print("gretel_client is not installed, skipping.")
            return
        if num_rows is None:
            num_rows = 1000
            warnings.warn("No 'num_rows' provided, defaulting to 1000.")

        config = read_model_config("synthetics/tabular-actgan")
        config["models"][0]["actgan"]["params"]["epochs"] = "auto"
        config["models"][0]["actgan"]["generate"]["num_records"] = num_rows

        model = self.project.create_model_obj(
            model_config=config, data_source=orig_df.to_pandas()
        )
        model.submit_cloud()
        print(
            f"Follow along with training in the console: {self.project.get_console_url()}"
        )
        poll(model, verbose=False)

        synthetic_df = bpd.read_pandas(
            pd.read_csv(model.get_artifact_link("data_preview"), compression="gzip")
        )
        return synthetic_df
