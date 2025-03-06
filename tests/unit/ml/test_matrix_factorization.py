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


# import pytest

from bigframes.ml import decomposition


def test_decomposition_mf_model():
    model = decomposition.MatrixFactorization(
        num_factors=16,
        feedback_type="implicit",
        user_col="user_id",
        item_col="item_col",
        rating_col="rating_col",
        l2_reg=9.83,
    )
    assert model.num_factors == 16
    assert model.feedback_type == "implicit"
    assert model.user_col == "user_id"
    assert model.item_col == "item_col"
    assert model.rating_col == "rating_col"


def test_decomposition_mf_feedback_type_explicit():
    model = decomposition.MatrixFactorization(
        num_factors=16,
        feedback_type="explicit",
        user_col="user_id",
        item_col="item_col",
        rating_col="rating_col",
        l2_reg=9.83,
    )
    assert model.feedback_type == "explicit"


# test_decomposition_mf_invalid_feedback_type_raises


def test_decomposition_mf_num_factors_low():
    model = decomposition.MatrixFactorization(
        num_factors=0,
        feedback_type="explicit",
        user_col="user_id",
        item_col="item_col",
        rating_col="rating_col",
        l2_reg=9.83,
    )
    assert model.num_factors == 0


#   test_decomposition_mf_negative_num_factors_raises

# def test_decomposition_mf_invalid_num_factors_raises():
#     num_factors = 0.5
#     with pytest.raises(TypeError):
#         decomposition.MatrixFactorization(
#             num_factors=num_factors,
#             feedback_type="explicit",
#             user_col="user_id",
#             item_col="item_col",
#             rating_col="rating_col",
#             l2_reg=9.83,
#         )


# def test_decomposition_mf_invalid_user_col_raises():
#     with pytest.raises(TypeError):
#         decomposition.MatrixFactorization(
#             num_factors=16,
#             feedback_type="explicit",
#             user_col=123,
#             item_col="item_col",
#             rating_col="rating_col",
#             l2_reg=9.83,
#         )


# def test_decomposition_mf_invalid_item_col_raises():
#     with pytest.raises(TypeError):
#         decomposition.MatrixFactorization(
#             num_factors=16,
#             feedback_type="explicit",
#             user_col="user_col",
#             item_col=123,
#             rating_col="rating_col",
#             l2_reg=9.83,
#         )
