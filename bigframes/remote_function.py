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

import functools
import inspect
import itertools
import json
import logging
import os
import random
import shutil
import string
import sys
import tempfile
import textwrap
import time

import cloudpickle
import google.cloud.bigquery as bigquery
import google.cloud.functions_v2 as functions_v2
from ibis.backends.bigquery.compiler import compiles
from ibis.backends.bigquery.datatypes import ibis_type_to_bigquery_type
import ibis.expr.operations as ops
import ibis.expr.rules as rlz

# TODO(shobs): Pick up project, location and dataset from Session (session.py)
gcp_project_id = "bigframes-dev"
cloud_function_region = "us-central1"
bq_location = "us-central1"
bq_dataset = "test_us_central1"
bq_client = bigquery.Client(project=gcp_project_id)
bq_connection_id = "bigframes-rf-conn"
wait_seconds = 90

# Protocol version 4 is available in python version 3.4 and above
# https://docs.python.org/3/library/pickle.html#data-stream-format
pickle_protocol_version = 4

# TODO(shobs): Change the min log level to INFO after the development stabilizes
# before June 2023
logging.basicConfig(
    level=logging.DEBUG, format="[%(levelname)s][%(asctime)s][%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)


def create_bq_remote_function(
    input_args, input_types, output_type, endpoint, bq_function_name
):
    """Create a BigQuery remote function given the artifacts of a user defined
    function and the http endpoint of a corresponding cloud function."""
    # Command to show details of an existing BQ connection
    command_show = f"bq show --connection --format=json {gcp_project_id}.{bq_location}.{bq_connection_id}"

    # Command to list existing BQ connections
    command_ls = (
        f"bq ls --connection --project_id={gcp_project_id} --location={bq_location}"
    )

    # TODO(shobs): The below is passing on cloudtop with user credentials but
    # failing in kokoro which runs with service account credentials.
    #   ERROR: (gcloud.services.enable) PERMISSION_DENIED: Permission denied to
    #   enable service [bigqueryconnection.googleapis.com]
    # which suggests that the service account doesn't have enough privilege.
    # For now enabled the API via cloud console, but we should revisit whether
    # this needs to be automated for the BigFrames end user.
    # log("Making sure BigQuery Connection API is enabled")
    # if os.system("gcloud services enable bigqueryconnection.googleapis.com"):
    #    raise ValueError("Failed to enable BigQuery Connection API")

    logger.info("List of existing connections")
    if os.system(command_ls):
        raise ValueError("Failed to list bq connections")

    # If the intended connection does not exist then create it
    connector_exists = os.system(f"{command_show} 2>&1 >/dev/null") == 0
    if connector_exists:
        logger.info(f"Connector {bq_connection_id} already exists")
    else:
        # TODO(shobs): Find a more structured way of doing it than invoking CLI
        # Create BQ connection
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#create_a_connection
        command_mk = (
            "bq mk --connection"
            + f" --location={bq_location}"
            + f" --project_id={gcp_project_id}"
            + " --connection_type=CLOUD_RESOURCE"
            + f" {bq_connection_id}"
        )
        logger.info(f"Creating BQ connection: {command_mk}")
        if os.system(command_mk):
            raise ValueError("Failed to make bq connection")

        logger.info("List of connections after creating")
        if os.system(command_ls):
            raise ValueError("Failed to list bq connections")

        # Fetch the service account id of the connection
        # bq show --connection outputs the service account id
        # TODO(shobs): again, we are parsing shell command output, instead we should
        # be using more structured means, i.e. python/REST/grpc APIs
        logger.info(f"Fetching service account id for connection {bq_connection_id}")
        # TODO(shobs): Really fragile, try a better way to get the service account id
        bq_connection_details = os.popen(command_show).read()
        service_account_id = json.loads(bq_connection_details)["cloudResource"][
            "serviceAccountId"
        ]
        logger.info(
            f"connectionId: ({bq_connection_id}), serviceAccountId: ({service_account_id})"
        )

        # Set up access on the newly created BQ connection
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#grant_permission_on_function
        # We would explicitly wait for 60+ seconds for the IAM binding to take effect
        command_iam = (
            f"gcloud projects add-iam-policy-binding {gcp_project_id}"
            + f' --member="serviceAccount:{service_account_id}"'
            + ' --role="roles/run.invoker"'
        )
        logger.info(f"Setting up IAM binding on the BQ connection: {command_iam}")
        if os.system(command_iam):
            raise ValueError("Failed to set up iam for the bq connection")

        logger.info(f"Waiting {wait_seconds} seconds for IAM to take effect..")
        time.sleep(wait_seconds)

    # Create BQ function
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#create_a_remote_function_2
    bq_function_args = []
    bq_function_return_type = ibis_type_to_bigquery_type(output_type)
    # We are expecting the input type annotations to be 1:1 with the input args
    for idx, name in enumerate(input_args):
        bq_function_args.append(
            f"{name} {ibis_type_to_bigquery_type(input_types[idx])}"
        )
    create_function_ddl = f"""
CREATE OR REPLACE FUNCTION `{gcp_project_id}.{bq_dataset}`.{bq_function_name}({','.join(bq_function_args)})
RETURNS {bq_function_return_type}
REMOTE WITH CONNECTION `{gcp_project_id}.{bq_location}.{bq_connection_id}`
OPTIONS (
  endpoint = "{endpoint}"
)"""
    command_rf = f"bq query --use_legacy_sql=false '{create_function_ddl}'"
    logger.info(f"Creating BQ remote function: {command_rf}")
    if os.system(command_rf):
        raise ValueError("Failed to make bq remote function")

    logger.info(
        f"Created remote function `{gcp_project_id}.{bq_dataset}`.{bq_function_name}"
    )


def get_cloud_function_endpoint(name):
    """Get the http endpoint of a cloud function if it exists."""
    client = functions_v2.FunctionServiceClient()
    parent = f"projects/{gcp_project_id}/locations/{cloud_function_region}"
    request = functions_v2.ListFunctionsRequest(parent=parent)
    page_result = client.list_functions(request=request)
    expected_cf_name = parent + f"/functions/{name}"
    for response in page_result:
        if response.name == expected_cf_name:
            return response.service_config.uri
    return ""


def generate_udf_code(def_, dir):
    """Generate serialized bytecode using cloudpickle given a udf."""
    udf_code_file_name = "udf.py"
    udf_bytecode_file_name = "udf.cloudpickle"

    # original code
    # TODO(shobs): Let's assume it's a simple user defined function with a
    # single decorator that happens to be the remote function decorator itself.
    # In case of multiple decorators the cloud function source code should
    # be generated with the decorators other than the remote function decorator.
    udf_lines = list(
        itertools.dropwhile(
            lambda line: (not line.lstrip().startswith("def ")),
            inspect.getsourcelines(def_)[0],
        )
    )
    udf_code = textwrap.dedent("".join(udf_lines))
    udf_code_file_path = os.path.join(dir, udf_code_file_name)
    with open(udf_code_file_path, "w") as f:
        f.write(udf_code)

    # serialized bytecode
    udf_bytecode_file_path = os.path.join(dir, udf_bytecode_file_name)
    with open(udf_bytecode_file_path, "wb") as f:
        cloudpickle.dump(def_, f, protocol=pickle_protocol_version)

    return udf_code_file_name, udf_bytecode_file_name


def generate_cloud_function_main_code(def_, dir):
    """Get main.py code for the cloud function for the given user defined function."""

    # Pickle the udf with all its dependencies
    udf_code_file, udf_bytecode_file = generate_udf_code(def_, dir)
    handler_func_name = "udf_http"

    # We want to build a cloud function that works for BQ remote functions,
    # where we receive `calls` in json which is a batch of rows from BQ SQL.
    # The number and the order of values in each row is expected to exactly
    # match to the number and order of arguments in the udf , e.g. if the udf is
    #   def foo(x: int, y: str):
    #     ...
    # then the http request body could look like
    # {
    #   ...
    #   "calls" : [
    #     [123, "hello"],
    #     [456, "world"]
    #   ]
    #   ...
    # }
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#input_format
    code_template = textwrap.dedent(
        """\
    import cloudpickle
    import json

    # original udf code is in {udf_code_file}
    # serialized udf code is in {udf_bytecode_file}
    with open("{udf_bytecode_file}", "rb") as f:
      udf = cloudpickle.load(f)

    def {handler_func_name}(request):
      request_json = request.get_json(silent=True)
      print("[debug] received json request: " + str(request_json))
      calls = request_json["calls"]
      replies = []
      for call in calls:
        reply = udf(*call)
        replies.append(reply)
      return_json = json.dumps({{"replies" : replies}})
      return return_json
    """
    )

    code = code_template.format(
        udf_code_file=udf_code_file,
        udf_bytecode_file=udf_bytecode_file,
        handler_func_name=handler_func_name,
    )

    main_py = os.path.join(dir, "main.py")
    with open(main_py, "w") as f:
        f.write(code)
    logger.debug(f"Wrote {os.path.abspath(main_py)}:\n{open(main_py).read()}")

    return handler_func_name


def generate_cloud_function_code(def_, dir):
    """Generate the cloud function code for a given user defined function."""

    # requirements.txt
    requirements = ["cloudpickle >= 2.1.0"]
    requirements_txt = os.path.join(dir, "requirements.txt")
    with open(requirements_txt, "w") as f:
        f.write("\n".join(requirements))

    # main.py
    entry_point = generate_cloud_function_main_code(def_, dir)
    return entry_point


def create_cloud_function(def_, cf_name):
    """Create a cloud function from the given user defined function."""

    # Display existing cloud functions before creation
    logger.info("Existing cloud functions")
    os.system(f"gcloud functions list --project={gcp_project_id}")

    # Build and deploy folder structure containing cloud function
    with tempfile.TemporaryDirectory() as dir:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

        entry_point = generate_cloud_function_code(def_, dir)

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

        # deploy/redeploy the cloud function
        # TODO(shobs): Figure out a way to skip this step if a cloud function
        # already exists with the same name and source code
        command = (
            "gcloud functions deploy"
            + f" {cf_name} --gen2"
            + f" --runtime={python_version}"
            + f" --project={gcp_project_id}"
            + f" --region={cloud_function_region}"
            + f" --source={dir}"
            + f" --entry-point={entry_point}"
            + " --trigger-http"
        )

        # If the cloud function is being created for the first time, then let's
        # make it not allow unauthenticated calls. If it was previously created
        # then this invocation will update it, in which case do not touch that
        # aspect and let the previous policy hold. The reason we do this is to
        # avoid an IAM permission needed to update the invocation policy.
        # For example, when a cloud function is being created for the first
        # time, i.e.
        # $ gcloud functions deploy python-foo-http --gen2 --runtime=python310
        #       --region=us-central1
        #       --source=/source/code/dir
        #       --entry-point=foo_http
        #       --trigger-http
        #       --no-allow-unauthenticated
        # It works. When an invocation of the same command is done for the
        # second time, it may run into an error like:
        # ERROR: (gcloud.functions.deploy) PERMISSION_DENIED: Permission
        # 'run.services.setIamPolicy' denied on resource
        # 'projects/my_project/locations/us-central1/services/python-foo-http' (or resource may not exist)
        # But when --no-allow-unauthenticated is omitted then it goes through.
        # It suggests that in the second invocation the command is trying to set
        # the IAM policy of the service, and the user running BigFrames may not
        # have privilege to do so, so better avoid this if we can.
        if get_cloud_function_endpoint(cf_name):
            logger.info(f"Updating existing cloud function: {command}")
        else:
            command = f"{command} --no-allow-unauthenticated"
            logger.info(f"Creating new cloud function: {command}")
        if os.system(command):
            raise ValueError("Failed at gcloud functions deploy")

    # Display existing cloud functions after creation
    logger.info("Existing cloud functions")
    os.system(f"gcloud functions list --project={gcp_project_id}")

    # Fetch the endpoint of the just created function
    endpoint = get_cloud_function_endpoint(cf_name)
    if not endpoint:
        raise ValueError("Couldn't fetch the http endpoint")

    logger.info(f"Successfully created cloud function {cf_name} with uri ({endpoint})")
    return (cf_name, endpoint)


def get_cloud_function_name(def_, uniq_suffix=None):
    """Get the name of the cloud function."""
    cf_name = f'bigframes-{def_.__name__.replace("_", "-")}'
    if uniq_suffix:
        cf_name = f"{cf_name}-{uniq_suffix}"
    return cf_name


def get_remote_function_name(def_, uniq_suffix=None):
    """Get the name for the BQ remote function."""
    bq_rf_name = f"bigframes_{def_.__name__}"
    if uniq_suffix:
        bq_rf_name = f"{bq_rf_name}_{uniq_suffix}"
    return bq_rf_name


def provision_bq_remote_function(
    def_, input_types, output_type, cloud_function_name, remote_function_name
):
    """Provision a BigQuery remote function."""
    _, endpoint = create_cloud_function(def_, cloud_function_name)
    input_args = inspect.getargs(def_.__code__).args
    if len(input_args) != len(input_types):
        raise ValueError("Exactly one type should be provided for every input arg.")
    create_bq_remote_function(
        input_args, input_types, output_type, endpoint, remote_function_name
    )
    return remote_function_name


def bq_remote_function_exists(remote_function_name):
    """Check whether a remote function already exists for the udf."""
    routines = bq_client.list_routines(f"{gcp_project_id}.{bq_dataset}")
    for routine in routines:
        if routine.reference.routine_id == remote_function_name:
            return True
    return False


def check_tools_and_permissions():
    """Check if the necessary tools and permissions are in place for creating remote function"""
    # gcloud CLI comes with bq CLI and they are required for creating google
    # cloud function and BigQuery remote function respectively
    if not shutil.which("gcloud"):
        raise ValueError(
            "gcloud tool not installed, install it from https://cloud.google.com/sdk/docs/install"
        )

    # TODO(shobs): Check for permissions too
    # I (shobs) tried the following method
    # $ gcloud asset search-all-iam-policies \
    #   --format=json \
    #   --scope=projects/{gcp_project_id} \
    #   --query='policy.role.permissions:cloudfunctions.functions.create'
    # as a proxy to all the privilges necessary to create cloud function
    # https://cloud.google.com/functions/docs/reference/iam/roles#cloudfunctions.developer
    # but that itself required the runner to have the permission to enable
    # `cloudasset.googleapis.com`


def provision_bq_remote_function_if_needed(
    def_, input_types, output_type, uniq_suffix=None
):
    """Provision a BigQuery remote function if it does not already exist."""
    remote_function_name = get_remote_function_name(def_, uniq_suffix)
    if not bq_remote_function_exists(remote_function_name):
        logger.info(f"Provisioning new remote function {def_.__name__} ...")
        check_tools_and_permissions()
        cloud_function_name = get_cloud_function_name(def_, uniq_suffix)
        return provision_bq_remote_function(
            def_, input_types, output_type, cloud_function_name, remote_function_name
        )
    else:
        logger.info(f"Remote function {def_.__name__} already exists, reusing ...")
    return remote_function_name


def get_remote_function_locations(bq_location):
    """Get BQ location and cloud functions region given a BQ client."""
    # TODO(shobs, b/274647164): Find the best way to determine default location.
    # For now let's assume that if no BQ location is set in the client then it
    # defaults to US multi region
    bq_location = bq_client.location.lower() if bq_client.location else "us"

    # Cloud function should be in the same region as the bigquery remote function
    cloud_function_region = bq_location

    # BigQuery has multi region but cloud functions does not.
    # Any region in the multi region that supports cloud functions should work
    # https://cloud.google.com/functions/docs/locations
    if bq_location == "us":
        cloud_function_region = "us-central1"
    elif bq_location == "eu":
        cloud_function_region = "europe-west1"

    return bq_location, cloud_function_region


#
# Inspired by @udf decorator implemented in ibis-bigquery package
# https://github.com/ibis-project/ibis-bigquery/blob/main/ibis_bigquery/udf/__init__.py
# which has moved as @js to the ibis package
# https://github.com/ibis-project/ibis/blob/master/ibis/backends/bigquery/udf/__init__.py
def remote_function(
    input_types,
    output_type,
    bigquery_client: bigquery.Client,
    dataset: str,
    bigquery_connection: str,
    reuse: bool = True,
):
    """Decorator to turn a user defined function into a BigQuery remote function.

    Parameters
    ----------
    input_types : list(ibis.expr.datatypes)
        List of input data types in the user defined function.
    output_type : ibis.expr.datatypes
        Data type of the output in the user defined function.
    bigquery_client : google.cloud.bigquery.client.Client
        Client to use for BigQuery operations.
    dataset : str
        Dataset to use to create a BigQuery function. It should be in
        `project_id.dataset_name` format
    bigquery_connection : str
        Name of the BigQuery connection. It should be pre created in the same
        location as the `bigquery_client.location`.
    reuse : bool
        Reuse the remote function if already exists.
        `True` by default, which will result in reusing an existing remote
        function (if any) that was previously created for the same udf.
        Setting it to false would force creating a unique remote function.

    Prerequisites
    -------------
    Please make sure following is setup before using this API:

    1. Have below APIs enabled for your project:
        a. Cloud Build API
        b. Artifact Registry API
        c. Cloud Functions API
        d. BigQuery Connection API

        This can be done from the cloud console (change PROJECT_ID to yours):
            https://console.cloud.google.com/apis/enableflow?apiid=cloudbuild.googleapis.com,artifactregistry.googleapis.com,cloudfunctions.googleapis.com,bigqueryconnection.googleapis.com&project=PROJECT_ID
        Or from the gcloud CLI:
            gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com cloudfunctions.googleapis.com bigqueryconnection.googleapis.com

    2. Have a BigQuery connection created and IAM role set up.
        a. To create a connection, follow https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#create_a_connection
        b. To set up IAM, follow https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#grant_permission_on_function
            Alternatively, the IAM could also be setup via the gcloud CLI:
            gcloud projects add-iam-policy-binding PROJECT_ID --member=CONNECTION_SERVICE_ACCOUNT_ID --role="roles/run.invoker"

    """

    # BQ remote function must be persisted, for which we need a dataset
    # https://cloud.google.com/bigquery/docs/reference/standard-sql/remote-functions#:~:text=You%20cannot%20create%20temporary%20remote%20functions.
    if not dataset:
        raise ValueError("Dataset must be provided to create BQ remote function")
    if not bigquery_connection:
        raise ValueError(
            "BigQuery connection muust be provided to be used in BQ remote function"
        )
    dataset_parts = dataset.split(".")
    if len(dataset_parts) != 2:
        raise ValueError(
            f'Expected a fully-qualified dataset ID in standard SQL format e.g. "project.dataset_id", got {dataset}'
        )

    # Set globals used in creating/probing GCP resources
    global gcp_project_id, cloud_function_region, bq_location, bq_dataset, bq_client, bq_connection_id
    bq_client = bigquery_client
    gcp_project_id, bq_dataset = dataset_parts
    bq_location, cloud_function_region = get_remote_function_locations(
        bq_client.location
    )
    bq_connection_id = bigquery_connection

    uniq_suffix = None
    if not reuse:
        uniq_suffix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=8)
        )

    def wrapper(f):
        if not callable(f):
            raise TypeError("f must be callable, got {}".format(f))

        signature = inspect.signature(f)
        parameter_names = signature.parameters.keys()

        # TODO(shobs): Check that input_types maps 1:1 with the udf params and
        # each of the provided one is supported in the current implementation
        rf_node_fields = {
            name: rlz.value(type) for name, type in zip(parameter_names, input_types)
        }

        try:
            rf_node_fields["output_type"] = rlz.shape_like("args", dtype=output_type)
        except TypeError:
            rf_node_fields["output_dtype"] = property(lambda _: output_type)
            rf_node_fields["output_shape"] = rlz.shape_like("args")

        rf_name = provision_bq_remote_function_if_needed(
            f, input_types, output_type, uniq_suffix
        )
        rf_fully_qualified_name = f"`{gcp_project_id}.{bq_dataset}`.{rf_name}"
        rf_node = type(rf_fully_qualified_name, (ops.ValueOp,), rf_node_fields)

        @compiles(rf_node)
        def compiles_rf_node(t, op):
            return "{}({})".format(
                rf_node.__name__, ", ".join(map(t.translate, op.args))
            )

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            node = rf_node(*args, **kwargs)
            return node.to_expr()

        wrapped.__signature__ = signature
        return wrapped

    return wrapper
