from bigframes.ml import core, utils


def json_extract_array(X):
    (X,) = utils.convert_to_dataframe(X)
    if len(X.columns) != 1:
        raise ValueError("Inputs X can only contain 1 column.")

    base_bqml = core.BaseBqml(session=X._session)
    return base_bqml.json_extract_array(X, "json_array")
