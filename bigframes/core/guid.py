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

_GUID_COUNTER = 0


def generate_guid(prefix="col_"):
    global _GUID_COUNTER
    _GUID_COUNTER += 1
    return f"bfuid_{prefix}{_GUID_COUNTER}"


class SequentialUIDGenerator:
    """
    Generates sequential-like UIDs with multiple prefixes, e.g., "t0", "t1", "c0", "t2", etc.
    """

    def __init__(self):
        self.prefix_counters = {}

    def generate_sequential_uid(self, prefix: str) -> str:
        """Generates a sequential UID with specified prefix."""
        if prefix not in self.prefix_counters:
            self.prefix_counters[prefix] = 0

        uid = f"{prefix}{self.prefix_counters[prefix]}"
        self.prefix_counters[prefix] += 1
        return uid
