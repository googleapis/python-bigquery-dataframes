from typing import Union

from bigframes.dataframe import DataFrame
from bigframes.series import Series

from .ml import BaseEstimator


class LinearModelBase(BaseEstimator):
    """
    Implementation of BQML's "Generalized linear models" class of models.

    input_label_cols is determined by .fit(..), the rest of the parameters
    match the BQML spec:
        https://cloud.google.com/bigquery-ml/docs/reference/standard-sql/bigqueryml-syntax-create-glm

    TODO: Should we copy SKLearn's parameter validation system?
        https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/linear_model/_logistic.py#L1068
        It's not clear how to handle string enum parameters in python in a way that's easy to use :(
        We could use an Enum class and convert it to the BQML string internally, but having to import
        the enum and access its properties seems less user friendly that just passing the string value
        and would be less consistent with sklearn

    Valid parameters for all linear models:
        MODEL_TYPE={'LINEAR_REG' | 'LOGISTIC_REG'},
        INPUT_LABEL_COLS=string_array,
        OPTIMIZE_STRATEGY={'AUTO_STRATEGY' | 'BATCH_GRADIENT_DESCENT' | 'NORMAL_EQUATION'},
        L1_REG=float64_value,
        L2_REG=float64_value,
        MAX_ITERATIONS=int64_value,
        LEARN_RATE_STRATEGY={'LINE_SEARCH' | 'CONSTANT'},
        LEARN_RATE=float64_value,
        EARLY_STOP={TRUE | FALSE},
        MIN_REL_PROGRESS=float64_value,
        DATA_SPLIT_METHOD={'AUTO_SPLIT' | 'RANDOM' | 'CUSTOM' | 'SEQ' | 'NO_SPLIT'},
        DATA_SPLIT_EVAL_FRACTION=float64_value,
        DATA_SPLIT_COL=string_value,
        LS_INIT_LEARN_RATE=float64_value,
        WARM_START={TRUE | FALSE},
        AUTO_CLASS_WEIGHTS={TRUE | FALSE},
        CLASS_WEIGHTS=struct_array,
        ENABLE_GLOBAL_EXPLAIN={TRUE | FALSE},
        CALCULATE_P_VALUES={TRUE | FALSE},
        FIT_INTERCEPT={TRUE | FALSE},
        CATEGORY_ENCODING_METHOD={'ONE_HOT_ENCODING`, 'DUMMY_ENCODING'}
    """

    def __init__(
        self,
        optimize_strategy=None,
        l1_reg=None,
        l2_reg=None,
        max_iterations=None,
        learn_rate_strategy=None,
        learn_rate=None,
        early_stop=None,
        min_rel_progress=None,
        data_split_method=None,
        data_split_eval_fraction=None,
        data_split_col=None,
        ls_init_learn_rate=None,
        warm_start=None,
        auto_class_weights=None,
        class_weights=None,
        enable_global_explain=None,
        calculate_p_values=None,
        fit_intercept=None,
        category_encoding_method=None,
    ):
        self.model_name = "unimplemented_model_name"  # Should this be part of mlops? Temporary generated name?
        self.model_type = None  # Must be set by subclass

        self.optimize_strategy = optimize_strategy
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_iterations = max_iterations
        self.learn_rate_strategy = learn_rate_strategy
        self.learn_rate = learn_rate
        self.early_stop = early_stop
        self.min_rel_progress = min_rel_progress
        self.data_split_method = data_split_method
        self.data_split_eval_fraction = data_split_eval_fraction
        self.data_split_col = data_split_col
        self.ls_init_learn_rate = ls_init_learn_rate
        self.warm_start = warm_start
        self.auto_class_weights = auto_class_weights
        self.class_weights = class_weights
        self.enable_global_explain = enable_global_explain
        self.calculate_p_values = calculate_p_values
        self.fit_intercept = fit_intercept
        self.category_encoding_method = category_encoding_method

    def fit(self, train_x: Union[DataFrame, Series], train_y: Union[DataFrame, Series]):
        if self.model_type not in ("linear_reg", "logistic_reg"):
            raise ValueError("Model type must be specified")

        # TODO: Implement
        # BQML will need train_x and train_y to be combined into one projection. The
        # columns in train_y will then be used to populate INPUT_LABEL_COLS in the
        # BQML model options
        # We will create the model immediately and retun when it has finished training
        # (If we decide later that this should be lazy or async we can revisit this)
        print(train_x)
        print(train_y)
        print("Magically fitting to your data!")

    def predict(self, test_x: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
        if self.model_type not in ("linear_reg", "logistic_reg"):
            raise ValueError("Model type must be specified")

        # TODO: Implement
        # BQML will run prediction immediately and return results when finished
        # (If we decide later that this should be lazy or async we can revisit this)
        print("**casts ML magic on your data**")

        return test_x

    # TODO: Design - this may be one place where it is worth differing from SKLearn
    # def evaluate(self, test_x -> Union[DataFrame, Series]):
    #    run BQML model evaluation and return stats


# Even though these are thin wrappers of the generic BQML Linear Model, we provide
# them as subclasses for familiarity


class LinearRegression(LinearModelBase):
    """BQML's linear regression model"""

    model_type = "linear_reg"


class LogisticRegression(LinearModelBase):
    """BQML's logistic regression model"""

    model_type = "logistic_reg"
