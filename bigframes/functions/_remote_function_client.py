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


from __future__ import annotations

import inspect
import logging
import os
import random
import shutil
import string
import sys
import tempfile
from typing import cast, Tuple, TYPE_CHECKING

import requests

from bigframes import constants
import bigframes.functions.remote_function_template

if TYPE_CHECKING:
    from bigframes.session import Session

import google.api_core.exceptions
import google.api_core.retry
from google.cloud import bigquery, functions_v2

from . import _utils

logger = logging.getLogger(__name__)


class RemoteFunctionClient:
    # Wait time (in seconds) for an IAM binding to take effect after creation
    _iam_wait_seconds = 120

    def __init__(
        self,
        gcp_project_id,
        cloud_function_region,
        cloud_functions_client,
        bq_location,
        bq_dataset,
        bq_client,
        bq_connection_id,
        bq_connection_manager,
        cloud_function_service_account,
        cloud_function_kms_key_name,
        cloud_function_docker_repository,
        *,
        session: Session,
    ):
        self._gcp_project_id = gcp_project_id
        self._cloud_function_region = cloud_function_region
        self._cloud_functions_client = cloud_functions_client
        self._bq_location = bq_location
        self._bq_dataset = bq_dataset
        self._bq_client = bq_client
        self._bq_connection_id = bq_connection_id
        self._bq_connection_manager = bq_connection_manager
        self._cloud_function_service_account = cloud_function_service_account
        self._cloud_function_kms_key_name = cloud_function_kms_key_name
        self._cloud_function_docker_repository = cloud_function_docker_repository
        self._session = session

    def create_bq_remote_function(
        self,
        input_args,
        input_types,
        output_type,
        endpoint,
        bq_function_name,
        max_batching_rows,
    ):
        """Create a BigQuery remote function given the artifacts of a user defined
        function and the http endpoint of a corresponding cloud function."""
        if self._bq_connection_manager:
            self._bq_connection_manager.create_bq_connection(
                self._gcp_project_id,
                self._bq_location,
                self._bq_connection_id,
                "run.invoker",
            )

        # Create BQ function
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#create_a_remote_function_2
        bq_function_args = []
        bq_function_return_type = output_type

        # We are expecting the input type annotations to be 1:1 with the input args
        for name, type_ in zip(input_args, input_types):
            bq_function_args.append(f"{name} {type_}")

        remote_function_options = {
            "endpoint": endpoint,
            "max_batching_rows": max_batching_rows,
        }

        remote_function_options_str = ", ".join(
            [
                f'{key}="{val}"' if isinstance(val, str) else f"{key}={val}"
                for key, val in remote_function_options.items()
                if val is not None
            ]
        )

        create_function_ddl = f"""
            CREATE OR REPLACE FUNCTION `{self._gcp_project_id}.{self._bq_dataset}`.{bq_function_name}({','.join(bq_function_args)})
            RETURNS {bq_function_return_type}
            REMOTE WITH CONNECTION `{self._gcp_project_id}.{self._bq_location}.{self._bq_connection_id}`
            OPTIONS ({remote_function_options_str})"""

        logger.info(f"Creating BQ remote function: {create_function_ddl}")

        # Make sure the dataset exists. I.e. if it doesn't exist, go ahead and
        # create it
        dataset = bigquery.Dataset(
            bigquery.DatasetReference.from_string(
                self._bq_dataset, default_project=self._gcp_project_id
            )
        )
        dataset.location = self._bq_location
        try:
            # This check does not require bigquery.datasets.create IAM
            # permission. So, if the data set already exists, then user can work
            # without having that permission.
            self._bq_client.get_dataset(dataset)
        except google.api_core.exceptions.NotFound:
            # This requires bigquery.datasets.create IAM permission
            self._bq_client.create_dataset(dataset, exists_ok=True)

        # TODO(swast): plumb through the original, user-facing api_name.
        _, query_job = self._session._start_query(create_function_ddl)
        logger.info(f"Created remote function {query_job.ddl_target_routine}")

    def get_cloud_function_fully_qualified_parent(self):
        "Get the fully qualilfied parent for a cloud function."
        return self._cloud_functions_client.common_location_path(
            self._gcp_project_id, self._cloud_function_region
        )

    def get_cloud_function_fully_qualified_name(self, name):
        "Get the fully qualilfied name for a cloud function."
        return self._cloud_functions_client.function_path(
            self._gcp_project_id, self._cloud_function_region, name
        )

    def get_remote_function_fully_qualilfied_name(self, name):
        "Get the fully qualilfied name for a BQ remote function."
        return f"{self._gcp_project_id}.{self._bq_dataset}.{name}"

    def get_cloud_function_endpoint(self, name):
        """Get the http endpoint of a cloud function if it exists."""
        fully_qualified_name = self.get_cloud_function_fully_qualified_name(name)
        try:
            response = self._cloud_functions_client.get_function(
                name=fully_qualified_name
            )
            return response.service_config.uri
        except google.api_core.exceptions.NotFound:
            pass
        return None

    def generate_cloud_function_code(
        self,
        def_,
        directory,
        *,
        input_types: Tuple[str],
        output_type: str,
        package_requirements=None,
        is_row_processor=False,
    ):
        """Generate the cloud function code for a given user defined function.

        Args:
            input_types (tuple[str]):
                Types of the input arguments in BigQuery SQL data type names.
            output_type (str):
                Types of the output scalar as a BigQuery SQL data type name.
        """

        # requirements.txt
        if package_requirements:
            requirements_txt = os.path.join(directory, "requirements.txt")
            with open(requirements_txt, "w") as f:
                f.write("\n".join(package_requirements))

        # main.py
        entry_point = bigframes.functions.remote_function_template.generate_cloud_function_main_code(
            def_,
            directory,
            input_types=input_types,
            output_type=output_type,
            is_row_processor=is_row_processor,
        )
        return entry_point

    def create_cloud_function(
        self,
        def_,
        cf_name,
        *,
        input_types: Tuple[str],
        output_type: str,
        package_requirements=None,
        timeout_seconds=600,
        max_instance_count=None,
        is_row_processor=False,
        vpc_connector=None,
        memory_mib=1024,
    ):
        """Create a cloud function from the given user defined function.

        Args:
            input_types (tuple[str]):
                Types of the input arguments in BigQuery SQL data type names.
            output_type (str):
                Types of the output scalar as a BigQuery SQL data type name.
        """

        # Build and deploy folder structure containing cloud function
        with tempfile.TemporaryDirectory() as directory:
            entry_point = self.generate_cloud_function_code(
                def_,
                directory,
                package_requirements=package_requirements,
                input_types=input_types,
                output_type=output_type,
                is_row_processor=is_row_processor,
            )
            archive_path = shutil.make_archive(directory, "zip", directory)

            # We are creating cloud function source code from the currently running
            # python version. Use the same version to deploy. This is necessary
            # because cloudpickle serialization done in one python version and
            # deserialization done in another python version doesn't work.
            # TODO(shobs): Figure out how to achieve version compatibility, specially
            # when pickle (internally used by cloudpickle) guarantees that:
            # https://docs.python.org/3/library/pickle.html#:~:text=The%20pickle%20serialization%20format%20is,unique%20breaking%20change%20language%20boundary.
            python_version = "python{}{}".format(
                sys.version_info.major, sys.version_info.minor
            )

            # Determine an upload URL for user code
            upload_url_request = functions_v2.GenerateUploadUrlRequest(
                kms_key_name=self._cloud_function_kms_key_name
            )
            upload_url_request.parent = self.get_cloud_function_fully_qualified_parent()
            upload_url_response = self._cloud_functions_client.generate_upload_url(
                request=upload_url_request
            )

            # Upload the code to GCS
            with open(archive_path, "rb") as f:
                response = requests.put(
                    upload_url_response.upload_url,
                    data=f,
                    headers={"content-type": "application/zip"},
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        "Failed to upload user code. code={}, reason={}, text={}".format(
                            response.status_code, response.reason, response.text
                        )
                    )

            # Deploy Cloud Function
            create_function_request = functions_v2.CreateFunctionRequest()
            create_function_request.parent = (
                self.get_cloud_function_fully_qualified_parent()
            )
            create_function_request.function_id = cf_name
            function = functions_v2.Function()
            function.name = self.get_cloud_function_fully_qualified_name(cf_name)
            function.build_config = functions_v2.BuildConfig()
            function.build_config.runtime = python_version
            function.build_config.entry_point = entry_point
            function.build_config.source = functions_v2.Source()
            function.build_config.source.storage_source = functions_v2.StorageSource()
            function.build_config.source.storage_source.bucket = (
                upload_url_response.storage_source.bucket
            )
            function.build_config.source.storage_source.object_ = (
                upload_url_response.storage_source.object_
            )
            function.build_config.docker_repository = (
                self._cloud_function_docker_repository
            )
            function.service_config = functions_v2.ServiceConfig()
            if memory_mib is not None:
                function.service_config.available_memory = f"{memory_mib}Mi"
            if timeout_seconds is not None:
                if timeout_seconds > 1200:
                    raise ValueError(
                        "BigQuery remote function can wait only up to 20 minutes"
                        ", see for more details "
                        "https://cloud.google.com/bigquery/quotas#remote_function_limits."
                    )
                function.service_config.timeout_seconds = timeout_seconds
            if max_instance_count is not None:
                function.service_config.max_instance_count = max_instance_count
            if vpc_connector is not None:
                function.service_config.vpc_connector = vpc_connector
            function.service_config.service_account_email = (
                self._cloud_function_service_account
            )
            function.kms_key_name = self._cloud_function_kms_key_name
            create_function_request.function = function

            # Create the cloud function and wait for it to be ready to use
            try:
                operation = self._cloud_functions_client.create_function(
                    request=create_function_request
                )
                operation.result()

                # Cleanup
                os.remove(archive_path)
            except google.api_core.exceptions.AlreadyExists:
                # If a cloud function with the same name already exists, let's
                # update it
                update_function_request = functions_v2.UpdateFunctionRequest()
                update_function_request.function = function
                operation = self._cloud_functions_client.update_function(
                    request=update_function_request
                )
                operation.result()

        # Fetch the endpoint of the just created function
        endpoint = self.get_cloud_function_endpoint(cf_name)
        if not endpoint:
            raise ValueError(
                f"Couldn't fetch the http endpoint. {constants.FEEDBACK_LINK}"
            )

        logger.info(
            f"Successfully created cloud function {cf_name} with uri ({endpoint})"
        )
        return endpoint

    def provision_bq_remote_function(
        self,
        def_,
        input_types,
        output_type,
        reuse,
        name,
        package_requirements,
        max_batching_rows,
        cloud_function_timeout,
        cloud_function_max_instance_count,
        is_row_processor,
        cloud_function_vpc_connector,
        cloud_function_memory_mib,
    ):
        """Provision a BigQuery remote function."""
        # Augment user package requirements with any internal package
        # requirements
        package_requirements = _utils._get_updated_package_requirements(
            package_requirements, is_row_processor
        )

        # Compute a unique hash representing the user code
        function_hash = _utils._get_hash(def_, package_requirements)

        # If reuse of any existing function with the same name (indicated by the
        # same hash of its source code) is not intended, then attach a unique
        # suffix to the intended function name to make it unique.
        uniq_suffix = None
        if not reuse:
            # use 4 digits as a unique suffix which should suffice for
            # uniqueness per session
            uniq_suffix = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=4)
            )

        # Derive the name of the cloud function underlying the intended BQ
        # remote function. Use the session id to identify the GCF for unnamed
        # functions. The named remote functions are treated as a persistant
        # artifacts, so let's keep them independent of session id, which also
        # makes their naming more stable for the same udf code
        session_id = None if name else self._session.session_id
        cloud_function_name = _utils.get_cloud_function_name(
            function_hash, session_id, uniq_suffix
        )
        cf_endpoint = self.get_cloud_function_endpoint(cloud_function_name)

        # Create the cloud function if it does not exist
        if not cf_endpoint:
            cf_endpoint = self.create_cloud_function(
                def_,
                cloud_function_name,
                input_types=input_types,
                output_type=output_type,
                package_requirements=package_requirements,
                timeout_seconds=cloud_function_timeout,
                max_instance_count=cloud_function_max_instance_count,
                is_row_processor=is_row_processor,
                vpc_connector=cloud_function_vpc_connector,
                memory_mib=cloud_function_memory_mib,
            )
        else:
            logger.info(f"Cloud function {cloud_function_name} already exists.")

        # Derive the name of the remote function
        remote_function_name = name
        if not remote_function_name:
            remote_function_name = _utils.get_remote_function_name(
                function_hash, self._session.session_id, uniq_suffix
            )
        rf_endpoint, rf_conn = self.get_remote_function_specs(remote_function_name)

        # Create the BQ remote function in following circumstances:
        # 1. It does not exist
        # 2. It exists but the existing remote function has different
        #    configuration than intended
        created_new = False
        if not rf_endpoint or (
            rf_endpoint != cf_endpoint or rf_conn != self._bq_connection_id
        ):
            input_args = inspect.getargs(def_.__code__).args
            if len(input_args) != len(input_types):
                raise ValueError(
                    "Exactly one type should be provided for every input arg."
                )
            self.create_bq_remote_function(
                input_args,
                input_types,
                output_type,
                cf_endpoint,
                remote_function_name,
                max_batching_rows,
            )

            created_new = True
        else:
            logger.info(f"Remote function {remote_function_name} already exists.")

        return remote_function_name, cloud_function_name, created_new

    def get_remote_function_specs(self, remote_function_name):
        """Check whether a remote function already exists for the udf."""
        http_endpoint = None
        bq_connection = None
        routines = self._bq_client.list_routines(
            f"{self._gcp_project_id}.{self._bq_dataset}"
        )
        try:
            for routine in routines:
                routine = cast(bigquery.Routine, routine)
                if routine.reference.routine_id == remote_function_name:
                    rf_options = routine.remote_function_options
                    if rf_options:
                        http_endpoint = rf_options.endpoint
                        bq_connection = rf_options.connection
                        if bq_connection:
                            bq_connection = os.path.basename(bq_connection)
                    break
        except google.api_core.exceptions.NotFound:
            # The dataset might not exist, in which case the http_endpoint doesn't, either.
            # Note: list_routines doesn't make an API request until we iterate on the response object.
            pass
        return (http_endpoint, bq_connection)
