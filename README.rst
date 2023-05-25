BigFrames
=========

BigFrames implements the Pandas dataframe API over top of a BigQuery session.

BigFrames also provides ``bigframes.ml``, which implements the Scikit-Learn API over top of BigQuery Machine Learning.

Quick start
-----------

Start a session

.. code-block:: python

    import bigframes
    session = bigframes.connect()

Initialize a dataframe for a BigQuery table

.. code-block:: python

    df = session.read_gbq("bigquery-public-data.ml_datasets.penguins")

    #Take a look
    df

Using the dataframe API
-----------------------

Start a session and initialize a dataframe for a BigQuery table

.. code-block:: python

    import bigframes
    session = bigframes.connect()

    df = session.read_gbq("bigquery-public-data.chicago_taxi_trips.taxi_trips")
    df

View the table schema

.. code-block:: python

    df.dtypes

Select a subset of columns

.. code-block:: python

    df = df[[
        "company",
        "trip_miles",
        "fare",
        "tips",
    ]]
    df

View the first ten values of a series

.. code-block:: python

    df['fare'].head(10)

Compute the mean of a series

.. code-block:: python

    df['fare'].mean()

Filter the dataframe

.. code-block:: python

    df[df['fare'] > 20.0]


Using the Remote Functions
--------------------------

BigFrames gives you the ability to turn your custom scalar functions into a BigQuery remote function.
It requires the GCP project to be set up appropriately and the user having sufficient privileges to use them.
One can find more details on it via `help` command.

.. code-block:: python

    help(bigframes.remote_function)

Define a custom function, and specify the intent to turn it into a remote function.
It requires a BigQuery connection. If the connection is not already created, BigFrames will
attempt to create one assuming the necessary APIs and IAM permissions are setup in the project.

.. code-block:: python

    @session.remote_function([float], float, bigquery_connection='bigframes-rf-conn')
    def get_capped_fare(fare):
        max_fare = 99.0
        return fare if fare <= max_fare else max_fare

Run the custom function on the BigFrames dataframe

.. code-block:: python

    df = df.assign(capped_fare=df['fare'].apply(get_capped_fare))
    df[['fare', 'capped_fare']].head(10)
