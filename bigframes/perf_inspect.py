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

import functools
import time

global_counter = 0

def runtime_logger(func):
    """Decorator to log the runtime of a function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global global_counter
        global_counter += 1
        prefix = "--" * global_counter

        start_time = time.monotonic()

        print(f"|{prefix}{func.__qualname__} started at {start_time:.2f} seconds")
        result = func(*args, **kwargs)
        end_time = time.monotonic()
        print(
            f"|{prefix}{func.__qualname__} ended at {end_time:.2f} seconds. "
            f"Runtime: {end_time - start_time:.2f} seconds"
        )
        global_counter -= 1
        return result

    return wrapper