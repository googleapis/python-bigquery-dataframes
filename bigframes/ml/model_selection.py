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

"""Functions for test/train split and model tuning. This module is styled after
Scikit-Learn's model_selection module:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection"""


from typing import List, Union

import bigframes


def train_test_split(
    *dataframes: bigframes.DataFrame,
    test_size: Union[float, None] = None,
    train_size: Union[float, None] = None,
    random_state: Union[int, None] = None
) -> List[bigframes.DataFrame]:
    """Splits dataframes into random train and test subsets

    Args:
        *dataframes: a sequence of BigFrames dataframes that can be joined on their indexes

        test_size: the proportion of the dataset to include in the test split. If None,
            this will default to the complement of train_size. If both are none, it will
            be set to 0.25

        train_size: the proportion of the dataset to include in the train split. If None,
            this will default to the complement of test_size.

        random_state: a seed to use for randomly choosing the rows of the split. If not
            set, a random split will be generated each time.

    Returns: a list of BigFrames DataFrame."""
    # TODO(bmil): Remove this once DataFrame.sample() is implemented
    session = dataframes[0]._block.expr._session
    pd_dataframes = [df.to_pandas() for df in dataframes]

    # TODO(bmil): Scikit-Learn throws an error when the dataframes don't have the same
    # number of rows. We probably want to do something similar, but as we're doing it
    # based on index, we probably need to do the following:
    # 1) Check if its the same underlying index (i.e. the dataframes are related)
    # 2) If (1) fails, dynamically check if they have the same values (i.e. that
    #    an outer join on all the indexes results in the same number of labels)
    # If (2) fails, error to the user
    # Alternatively, perhaps we should not use the index, as it may contain duplicates.
    # In this case we're going to need to move the implementation of this into the
    # dataframe, so we can compute the sampling based on the ordering.

    if test_size is None:
        if train_size is None:
            test_size = 0.25
        else:
            test_size = 1.0 - train_size
    if train_size is None:
        train_size = 1.0 - test_size

    test_index = (
        pd_dataframes[0].sample(frac=test_size, random_state=random_state).index
    )
    if train_size + test_size < 1.0:
        # when the training set isn't the complement of the test set, we need to split again
        # TODO(bmil): see if this is much slower than complicating DataFrame.sample to
        # handle two splits
        # TODO(bmil): the double split here includes the remainder of the test% in the train%
        # we should adjust this away so both splits work the same
        # TODO(bmil): pandas.sample seems to take the ceil of fract*num_rows, check that this
        # is consistent with sklearn
        frac = train_size / (1.0 - test_size)
        train_index = (
            pd_dataframes[0]
            .drop(test_index)
            .sample(frac=frac, random_state=random_state)
            .index
        )
    else:
        train_index = pd_dataframes[0].drop(test_index).index

    results = [
        df.loc[index] for df in pd_dataframes for index in (train_index, test_index)
    ]

    # TODO(bmil): remove this once DataFrame.sample is implemented
    return [session.read_pandas(df) for df in results]
