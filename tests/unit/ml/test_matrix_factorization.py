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

# import re

# import pytest

from bigframes.ml import decomposition


def test_decomposition_mf_num_factors():
    model = decomposition.MatrixFactorization(
        num_factors=16,
        feedback_type="explicit",
        user_col="user_id",
        item_col="item_col",
        rating_col="rating_col",
        l2_reg=9.83,
    )
    assert model.num_factors == 16


# def test_decomposition_mf_num_factors_invalid_raises():
#     # with pytest.raises(TypeError):
#     model = decomposition.MatrixFactorization(
#         num_factors=0.5,
#         feedback_type="explicit",
#         user_col="user_id",
#         item_col="item_col",
#         rating_col="rating_col",
#         l2_reg=9.83,
#     )
#     # passing test -> should raise error?
#     assert model.num_factors == 0.5


def test_decomposition_mf_feedback_type():
    model = decomposition.MatrixFactorization(
        num_factors=16,
        feedback_type="implicit",
        user_col="user_id",
        item_col="item_col",
        rating_col="rating_col",
        l2_reg=9.83,
    )
    assert model.feedback_type == "implicit"


# def test_decomposition_mf_feedback_type_raises():
#     model = decomposition.MatrixFactorization(
#         num_factors=16,
#         feedback_type="implexpl",
#         user_col="user_id",
#         item_col="item_col",
#         rating_col="rating_col",
#         l2_reg=9.83,
#     )
#     # passing test -> should raise error?
#     assert model.feedback_type == "implexpl"
