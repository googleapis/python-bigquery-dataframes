from bigframes.ml import LinearRegression, LogisticRegression


def test_linear_regression_parameters():
    model = LinearRegression(fit_intercept=True)
    assert model.get_params()["fit_intercept"]


def test_logistic_regression_repr():
    model = LogisticRegression()
    assert model.__repr__() == "I can't beleive it's not SKLearn!"
