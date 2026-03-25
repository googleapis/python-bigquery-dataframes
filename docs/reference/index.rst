API Reference
=============

The **BigQuery DataFrames (BigFrames) API Reference** documents the pandas-compatible and scikit-learn-compatible Python interfaces powered by BigQuery's distributed compute engine.

Designed to support the modern data stack, these APIs empower:

*   **Data Analysts** to write familiar pandas code for scalable data exploration, cleaning, and aggregation without hitting memory limits.
*   **Data Engineers** to build robust big data pipelines, leveraging advanced geospatial, array, and JSON functions native to BigQuery.
*   **Data Scientists** to train, evaluate, and deploy machine learning models directly on BigQuery using the ML modules, or integrate Generative AI via BigQuery ML and Gemini.

Use this reference to discover the classes, methods, and functions that make up the BigQuery DataFrames ecosystem.

.. autosummary::
    :toctree: api

    bigframes._config
    bigframes.bigquery
    bigframes.bigquery.ai
    bigframes.bigquery.ml
    bigframes.bigquery.obj
    bigframes.enums
    bigframes.exceptions
    bigframes.geopandas
    bigframes.pandas
    bigframes.pandas.api.typing
    bigframes.streaming

Pandas Extensions
~~~~~~~~~~~~~~~~~

BigQuery DataFrames provides extensions to pandas DataFrame objects.

.. autosummary::
    :toctree: api

    bigframes.extensions.pandas

ML APIs
~~~~~~~

BigQuery DataFrames provides many machine learning modules, inspired by
scikit-learn, enabling data scientists to quickly build, train, and deploy models
on large datasets natively within BigQuery.


.. autosummary::
    :toctree: api

    bigframes.ml
    bigframes.ml.cluster
    bigframes.ml.compose
    bigframes.ml.decomposition
    bigframes.ml.ensemble
    bigframes.ml.forecasting
    bigframes.ml.imported
    bigframes.ml.impute
    bigframes.ml.linear_model
    bigframes.ml.llm
    bigframes.ml.metrics
    bigframes.ml.model_selection
    bigframes.ml.pipeline
    bigframes.ml.preprocessing
    bigframes.ml.remote
