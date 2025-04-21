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
from __future__ import annotations

from typing import Optional

import pyarrow as pa

from bigframes.core import bigframe_node, rewrite
from bigframes.session import executor, semi_executor


class LocalScanExecutor(semi_executor.SemiExecutor):
    """
    Executes plans reducible to a arrow table scan.
    """

    def execute(
        self,
        plan: bigframe_node.BigFrameNode,
        ordered: bool,
        peek: Optional[int] = None,
    ) -> Optional[executor.ExecuteResult]:
        node = rewrite.try_reduce_to_local_scan(plan)
        if not node:
            return None

        # TODO: Can support some slicing, sorting
        def iterator_supplier():
            offsets_col = (
                node.offsets_col.sql if (node.offsets_col is not None) else None
            )
            arrow_table = node.local_data_source.to_pyarrow_table(
                offsets_col=offsets_col
            )
            if peek:
                arrow_table = arrow_table.slice(0, peek)
            for batch in arrow_table.to_batches():
                names = [item.id.sql for item in node.scan_list.items]
                arrays = [batch.column(item.source_id) for item in node.scan_list.items]
                yield pa.record_batch(arrays, names=names)

        return executor.ExecuteResult(
            arrow_batches=iterator_supplier,
            schema=plan.schema,
            query_job=None,
            total_bytes=None,
            total_rows=peek or node.local_data_source.metadata.row_count,
        )
