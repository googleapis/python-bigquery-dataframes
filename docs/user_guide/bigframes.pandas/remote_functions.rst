
Using the Remote Functions
==========================

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
