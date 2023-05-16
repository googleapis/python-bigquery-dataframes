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


Using the ML API
----------------

Start a session and initialize a dataframe for a BigQuery table

.. code-block:: python

    import bigframes
    session = bigframes.connect()

    df = session.read_gbq("bigquery-public-data.ml_datasets.penguins")
    df

Clean and prepare the data

.. code-block:: python

    # filter down to the data we want to analyze
    adelie_data = df[df.species == "Adelie Penguin (Pygoscelis adeliae)"]

    # drop the columns we don't care about
    adelie_data = adelie_data.drop(columns=["species"])

    # drop rows with nulls to get our training data
    training_data = adelie_data.dropna()

    # take a peek at the training data
    training_data

.. code-block:: python

    # pick feature columns and label column
    feature_columns = training_data[['island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'sex']]
    label_columns = training_data[['body_mass_g']]

    # also get the rows that we want to make predictions for (i.e. where the feature column is null)
    missing_body_mass = adelie_data[adelie_data.body_mass_g.isnull()]

Train and evaluate a linear regression model using the ML API

.. code-block:: python

    from bigframes.ml.linear_model import LinearRegression

    # as in scikit-learn, a newly created model is just a bundle of parameters
    # default parameters are fine here
    model = LinearRegression()

    # this will train a temporary model in BigQuery Machine Learning
    model.fit(feature_columns, label_columns)

    # check how the model performed, using the automatic test/training data split chosen by BQML
    model.score()

Make predictions using the model

.. code-block:: python

    model.predict(missing_body_mass)

Save the trained model to BigQuery, so we can load it later

.. code-block:: python

    model.to_gbq("bqml_tutorial.penguins_model", replace=True)
