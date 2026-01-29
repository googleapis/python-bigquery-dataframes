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

from unittest import mock

import pandas as pd

import bigframes
import bigframes.core.blocks as blocks


def test_block_split_rounding():
    # Setup a mock block with a specific shape
    mock_session = mock.create_autospec(spec=bigframes.Session)
    # Block.from_local needs a real-ish session for some things, but we can mock shape[0]

    # Let's use a real Block with local data for simplicity if possible
    df = pd.DataFrame({"a": range(29757)})
    block = blocks.Block.from_local(df, mock_session)

    # We need to mock the internal behavior of split or check the result sizes
    # Since split returns new Blocks, we can check their shapes if they are computed.
    # But split calls block.slice which calls block.expr.slice...

    # Instead of full execution, let's just test the rounding logic by mocking block.shape
    with mock.patch.object(
        blocks.Block, "shape", new_callable=mock.PropertyMock
    ) as mock_shape:
        mock_shape.return_value = (29757, 1)

        # We need to mock other things that split calls to avoid full execution
        with mock.patch.object(blocks.Block, "create_constant") as mock_create_constant:
            mock_create_constant.return_value = (block, "random_col")
            with mock.patch.object(
                blocks.Block, "promote_offsets"
            ) as mock_promote_offsets:
                mock_promote_offsets.return_value = (block, "offset_col")
                with mock.patch.object(
                    blocks.Block, "apply_unary_op"
                ) as mock_apply_unary_op:
                    mock_apply_unary_op.return_value = (block, "unary_col")
                    with mock.patch.object(
                        blocks.Block, "apply_binary_op"
                    ) as mock_apply_binary_op:
                        mock_apply_binary_op.return_value = (block, "binary_col")
                        with mock.patch.object(
                            blocks.Block, "order_by"
                        ) as mock_order_by:
                            mock_order_by.return_value = block
                            with mock.patch.object(blocks.Block, "slice") as mock_slice:
                                mock_slice.return_value = block

                                # Call split
                                block.split(fracs=(0.8, 0.2))

                                # Check calls to slice
                                # Expected sample_sizes with round():
                                # round(0.8 * 29757) = 23806
                                # round(0.2 * 29757) = 5951

                                calls = mock_slice.call_args_list
                                assert len(calls) == 2
                                assert calls[0].kwargs["start"] == 0
                                assert calls[0].kwargs["stop"] == 23806
                                assert calls[1].kwargs["start"] == 23806
                                assert calls[1].kwargs["stop"] == 23806 + 5951
