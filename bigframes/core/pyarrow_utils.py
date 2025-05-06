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
import itertools
from typing import Iterable, Iterator, Optional

import pyarrow as pa


def peek_batches(
    batch_iter: Iterable[pa.RecordBatch], max_bytes: int
) -> tuple[Iterator[pa.RecordBatch], Optional[tuple[pa.RecordBatch, ...]]]:
    """
    Try to peek a pyarrow batch iterable. If greater than max_bytes, give up.

    Will consume max_bytes + one batch of memory at worst.
    """
    batch_list = []
    current_bytes = 0
    for batch in batch_iter:
        batch_list.append(batch)
        current_bytes += batch.nbytes

        if current_bytes > max_bytes:
            return itertools.chain(batch_list, batch_iter), None

    return iter(batch_list), tuple(batch_list)
