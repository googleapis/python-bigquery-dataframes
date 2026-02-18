# Copyright 2026 Google LLC
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


def inherit_docs(source_class):
    """
    A class decorator that copies docstrings from source_class to the
    decorated class for any methods or attributes that match names.
    """

    def decorator(target_class):
        # 1. Steal the main class docstring if the target doesn't have one
        if not target_class.__doc__ and source_class.__doc__:
            target_class.__doc__ = source_class.__doc__

        # 2. Iterate over all attributes in the source class
        for name, source_item in vars(source_class).items():
            # Check if the target class has the same attribute
            if name in vars(target_class):
                target_item = getattr(target_class, name)

                # Only copy if the target doesn't have a docstring
                # and the source does
                if hasattr(target_item, "__doc__") and not target_item.__doc__:
                    if hasattr(source_item, "__doc__") and source_item.__doc__:
                        try:
                            # Use functools.update_wrapper or manual assignment
                            # for methods, properties, and static methods
                            target_item.__doc__ = source_item.__doc__
                        except AttributeError:
                            # Read-only attributes or certain built-ins
                            # might skip docstring assignment
                            pass

        return target_class

    return decorator
