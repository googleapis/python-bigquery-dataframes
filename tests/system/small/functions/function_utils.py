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

from bigframes.functions import _utils as bff_utils


def get_function_name(func, package_requirements=None, is_row_processor=False):
    """Get a bigframes function name for testing given a udf."""
    # Augment user package requirements with any internal package
    # requirements.
    package_requirements = bff_utils._get_updated_package_requirements(
        package_requirements, is_row_processor
    )

    # Compute a unique hash representing the user code.
    function_hash = bff_utils._get_hash(func, package_requirements)

    return f"bigframes_{function_hash}"
