.. BigQuery DataFrames documentation main file

Welcome to BigQuery DataFrames
==============================

**BigQuery DataFrames** (``bigframes``) provides a Pythonic interface for data analysis that scales to petabytes. It gives you the best of both worlds: the familiar API of **pandas** and **scikit-learn**, powered by the distributed computing engine of **BigQuery**.


Why BigQuery DataFrames?
------------------------

BigFrames allows you to process data where it lives. Instead of downloading massive datasets to your local machine, BigFrames translates your Python code into SQL and executes it across the BigQuery fleet.

* **Scalability:** Work with datasets that exceed local memory limits.
* **Efficiency:** Minimize data movement and leverage BigQuery's query optimizer.
* **Familiarity:** Use ``read_gbq``, ``merge``, ``groupby``, and ``pivot_table`` just like you do in pandas.
* **Integrated ML:** Access BigQuery ML (BQML) capabilities through a familiar estimator-based interface.


User Guide
----------

.. toctree::
    :maxdepth: 2

    user_guide/index

API reference
-------------

.. toctree::
    :maxdepth: 3

    reference/index
    supported_pandas_apis

Changelog
---------

For a list of all BigQuery DataFrames releases:

.. toctree::
    :maxdepth: 2

    changelog
