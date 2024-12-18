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

import math

import pandas as pd
import pytest

from bigframes.ml import model_selection
import bigframes.pandas as bpd


@pytest.mark.parametrize(
    "df_fixture",
    ("penguins_df_default_index", "penguins_df_null_index"),
)
def test_train_test_split_default_correct_shape(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
        ]
    ]
    y = df[["body_mass_g"]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    # even though the default seed is random, it should always result in this shape
    assert X_train.shape == (258, 3)
    assert X_test.shape == (86, 3)
    assert y_train.shape == (258, 1)
    assert y_test.shape == (86, 1)


def test_train_test_split_series_default_correct_shape(penguins_df_default_index):
    X = penguins_df_default_index[["species"]]
    y = penguins_df_default_index["body_mass_g"]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    assert isinstance(X_train, bpd.DataFrame)
    assert isinstance(X_test, bpd.DataFrame)
    assert isinstance(y_train, bpd.Series)
    assert isinstance(y_test, bpd.Series)

    # even though the default seed is random, it should always result in this shape
    assert X_train.shape == (258, 1)
    assert X_test.shape == (86, 1)
    assert y_train.shape == (258,)
    assert y_test.shape == (86,)


def test_train_test_double_split_correct_shape(penguins_df_default_index):
    X = penguins_df_default_index[
        [
            "species",
            "island",
            "culmen_length_mm",
        ]
    ]
    y = penguins_df_default_index[["body_mass_g"]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, train_size=0.4
    )

    # should have 20% in test, 40% in train, 40% dropped
    assert X_train.shape == (138, 3)
    assert X_test.shape == (69, 3)
    assert y_train.shape == (138, 1)
    assert y_test.shape == (69, 1)


def test_train_test_three_dataframes_correct_shape(penguins_df_default_index):
    A = penguins_df_default_index[
        [
            "species",
            "culmen_length_mm",
        ]
    ]
    B = penguins_df_default_index[
        [
            "island",
        ]
    ]
    C = penguins_df_default_index[["culmen_depth_mm", "body_mass_g"]]
    (
        A_train,
        A_test,
        B_train,
        B_test,
        C_train,
        C_test,
    ) = model_selection.train_test_split(A, B, C)

    assert A_train.shape == (258, 2)
    assert A_test.shape == (86, 2)
    assert B_train.shape == (258, 1)
    assert B_test.shape == (86, 1)
    assert C_train.shape == (258, 2)
    assert C_test.shape == (86, 2)


def test_train_test_split_seeded_correct_rows(
    session, penguins_pandas_df_default_index
):
    # Note that we're using `penguins_pandas_df_default_index` as this test depends
    # on a stable row order being present end to end
    # filter down to the chunkiest penguins, to keep our test code a reasonable size
    all_data = penguins_pandas_df_default_index[
        penguins_pandas_df_default_index.body_mass_g > 5500
    ]

    # Note that bigframes loses the index if it doesn't have a name
    all_data.index.name = "rowindex"

    df = session.read_pandas(all_data)

    X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
        ]
    ]
    y = df[["body_mass_g"]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, random_state=42
    )

    X_train_sorted = X_train.to_pandas().sort_index()
    X_test_sorted = X_test.to_pandas().sort_index()
    y_train_sorted = y_train.to_pandas().sort_index()
    y_test_sorted = y_test.to_pandas().sort_index()

    train_index: pd.Index = pd.Index(
        [
            144,
            146,
            168,
            183,
            186,
            217,
            221,
            225,
            237,
            240,
            244,
            245,
            257,
            260,
            262,
            263,
            264,
            266,
            267,
            268,
            290,
        ],
        dtype="Int64",
        name="rowindex",
    )
    test_index: pd.Index = pd.Index(
        [148, 161, 226, 269, 278, 289, 291], dtype="Int64", name="rowindex"
    )

    all_data.index.name = "_"

    assert (
        isinstance(X_train_sorted, pd.DataFrame)
        and isinstance(X_test_sorted, pd.DataFrame)
        and isinstance(y_train_sorted, pd.DataFrame)
        and isinstance(y_test_sorted, pd.DataFrame)
    )
    pd.testing.assert_frame_equal(
        X_train_sorted,
        all_data[
            [
                "species",
                "island",
                "culmen_length_mm",
            ]
        ].loc[train_index],
    )
    pd.testing.assert_frame_equal(
        X_test_sorted,
        all_data[
            [
                "species",
                "island",
                "culmen_length_mm",
            ]
        ].loc[test_index],
    )
    pd.testing.assert_frame_equal(
        y_train_sorted,
        all_data[
            [
                "body_mass_g",
            ]
        ].loc[train_index],
    )
    pd.testing.assert_frame_equal(
        y_test_sorted,
        all_data[
            [
                "body_mass_g",
            ]
        ].loc[test_index],
    )


@pytest.mark.parametrize(
    ("train_size", "test_size"),
    [
        (0.0, 0.5),
        (-0.5, 0.7),
        (0.5, 1.2),
        (0.6, 0.6),
    ],
)
def test_train_test_split_value_error(penguins_df_default_index, train_size, test_size):
    X = penguins_df_default_index[
        [
            "species",
            "island",
            "culmen_length_mm",
        ]
    ]
    y = penguins_df_default_index[["body_mass_g"]]
    with pytest.raises(ValueError):
        model_selection.train_test_split(
            X, y, train_size=train_size, test_size=test_size
        )


@pytest.mark.parametrize(
    "df_fixture",
    ("penguins_df_default_index", "penguins_df_null_index"),
)
def test_train_test_split_stratify(df_fixture, request):
    df = request.getfixturevalue(df_fixture)
    X = df[["species", "island", "culmen_length_mm",]].rename(
        columns={"species": "x_species"}
    )  # Keep "species" col just for easy checking. Rename to avoid conflicts.
    y = df[["species"]]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, stratify=df["species"]
    )

    # Original distribution is [152, 124, 68]. All the categories follow 75/25 split
    train_counts = pd.Series(
        [114, 93, 51],
        index=pd.Index(
            [
                "Adelie Penguin (Pygoscelis adeliae)",
                "Gentoo penguin (Pygoscelis papua)",
                "Chinstrap penguin (Pygoscelis antarctica)",
            ],
            name="species",
        ),
        dtype="Int64",
        name="count",
    )
    test_counts = pd.Series(
        [38, 31, 17],
        index=pd.Index(
            [
                "Adelie Penguin (Pygoscelis adeliae)",
                "Gentoo penguin (Pygoscelis papua)",
                "Chinstrap penguin (Pygoscelis antarctica)",
            ],
            name="species",
        ),
        dtype="Int64",
        name="count",
    )
    pd.testing.assert_series_equal(
        X_train["x_species"].rename("species").value_counts().to_pandas(),
        train_counts,
        check_index_type=False,
    )
    pd.testing.assert_series_equal(
        X_test["x_species"].rename("species").value_counts().to_pandas(),
        test_counts,
        check_index_type=False,
    )
    pd.testing.assert_series_equal(
        y_train["species"].value_counts().to_pandas(),
        train_counts,
        check_index_type=False,
    )
    pd.testing.assert_series_equal(
        y_test["species"].value_counts().to_pandas(),
        test_counts,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    "n_splits",
    (3, 5, 10),
)
def test_KFold_get_n_splits(n_splits):
    kf = model_selection.KFold(n_splits)
    assert kf.get_n_splits() == n_splits


@pytest.mark.parametrize(
    "df_fixture",
    ("penguins_df_default_index", "penguins_df_null_index"),
)
@pytest.mark.parametrize(
    "n_splits",
    (3, 5),
)
def test_KFold_split(df_fixture, n_splits, request):
    df = request.getfixturevalue(df_fixture)

    kf = model_selection.KFold(n_splits=n_splits)

    X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
        ]
    ]
    y = df["body_mass_g"]

    len_test_upper, len_test_lower = math.ceil(len(df) / n_splits), math.floor(
        len(df) / n_splits
    )
    len_train_upper, len_train_lower = (
        len(df) - len_test_lower,
        len(df) - len_test_upper,
    )

    for X_train, X_test, y_train, y_test in kf.split(X, y):  # type: ignore
        assert isinstance(X_train, bpd.DataFrame)
        assert isinstance(X_test, bpd.DataFrame)
        assert isinstance(y_train, bpd.Series)
        assert isinstance(y_test, bpd.Series)

        # Depend on the iteration, train/test can +-1 in size.
        assert (
            X_train.shape == (len_train_upper, 3)
            and y_train.shape == (len_train_upper,)
            and X_test.shape == (len_test_lower, 3)
            and y_test.shape == (len_test_lower,)
        ) or (
            X_train.shape == (len_train_lower, 3)
            and y_train.shape == (len_train_lower,)
            and X_test.shape == (len_test_upper, 3)
            and y_test.shape == (len_test_upper,)
        )


@pytest.mark.parametrize(
    "df_fixture",
    ("penguins_df_default_index", "penguins_df_null_index"),
)
@pytest.mark.parametrize(
    "n_splits",
    (3, 5),
)
def test_KFold_split_X_only(df_fixture, n_splits, request):
    df = request.getfixturevalue(df_fixture)

    kf = model_selection.KFold(n_splits=n_splits)

    X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
        ]
    ]

    len_test_upper, len_test_lower = math.ceil(len(df) / n_splits), math.floor(
        len(df) / n_splits
    )
    len_train_upper, len_train_lower = (
        len(df) - len_test_lower,
        len(df) - len_test_upper,
    )

    for X_train, X_test, y_train, y_test in kf.split(X, y=None):  # type: ignore
        assert isinstance(X_train, bpd.DataFrame)
        assert isinstance(X_test, bpd.DataFrame)
        assert y_train is None
        assert y_test is None

        # Depend on the iteration, train/test can +-1 in size.
        assert (
            X_train.shape == (len_train_upper, 3)
            and X_test.shape == (len_test_lower, 3)
        ) or (
            X_train.shape == (len_train_lower, 3)
            and X_test.shape == (len_test_upper, 3)
        )


def test_KFold_seeded_correct_rows(session, penguins_pandas_df_default_index):
    kf = model_selection.KFold(random_state=42)
    # Note that we're using `penguins_pandas_df_default_index` as this test depends
    # on a stable row order being present end to end
    # filter down to the chunkiest penguins, to keep our test code a reasonable size
    all_data = penguins_pandas_df_default_index[
        penguins_pandas_df_default_index.body_mass_g > 5500
    ]

    # Note that bigframes loses the index if it doesn't have a name
    all_data.index.name = "rowindex"

    df = session.read_pandas(all_data)

    X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
        ]
    ]
    y = df["body_mass_g"]
    X_train, X_test, y_train, y_test = next(kf.split(X, y))  # type: ignore

    X_train_sorted = X_train.to_pandas().sort_index()  # type: ignore
    X_test_sorted = X_test.to_pandas().sort_index()  # type: ignore
    y_train_sorted = y_train.to_pandas().sort_index()  # type: ignore
    y_test_sorted = y_test.to_pandas().sort_index()  # type: ignore

    train_index: pd.Index = pd.Index(
        [
            144,
            146,
            148,
            161,
            168,
            183,
            217,
            221,
            225,
            226,
            237,
            244,
            257,
            262,
            264,
            266,
            267,
            269,
            278,
            289,
            290,
            291,
        ],
        dtype="Int64",
        name="rowindex",
    )
    test_index: pd.Index = pd.Index(
        [186, 240, 245, 260, 263, 268], dtype="Int64", name="rowindex"
    )

    pd.testing.assert_index_equal(X_train_sorted.index, train_index)
    pd.testing.assert_index_equal(X_test_sorted.index, test_index)
    pd.testing.assert_index_equal(y_train_sorted.index, train_index)
    pd.testing.assert_index_equal(y_test_sorted.index, test_index)
