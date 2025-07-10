# This example demonstrates one of the most general usage of transforming raw
# BigQuery data into a processed table using dbt in BigFrames mode.
#
# Key defaults when using BigFrames in dbt:
# - The default materialization is 'table' unless specified otherwise.
# - The default timeout for the job is 3600 seconds (60 minutes).
# - If no runtime template is provided, dbt will automatically create and reuse
#   a default one.
#
# This code sample shows a basic pattern for reading a BigQuery public dataset,
# processing it using pandas-like operations, and outputting a cleaned table.


def model(dbt, session):
    # Optional: override settings from dbt_project.yml. When both are set,
    # dbt.config takes precedence over dbt_project.yml.
    # Use BigFrames mode to execute the Python model.
    dbt.config(submission_method="bigframes")

    # Define the BigQuery table path from which to read data.
    table = "bigquery-public-data.epa_historical_air_quality.temperature_hourly_summary"

    # Define the specific columns to select from the BigQuery table.
    columns = ["state_name", "county_name", "date_local", "time_local", "sample_measurement"]

    # Read data from the specified BigQuery table into a BigFrames DataFrame.
    # BigFrames allows you to interact with BigQuery tables using a pandas-like API.
    df = session.read_gbq(table, columns=columns)

    # Sort the DataFrame by the specified columns. This prepares the data for
    # `drop_duplicates` to ensure consistent duplicate removal.
    df = df.sort_values(columns).drop_duplicates(columns)

    # Group the DataFrame by 'state_name', 'county_name', and 'date_local'. For
    # each group, calculate the minimum and maximum of the 'sample_measurement'
    # column. The result will be a BigFrames DataFrame with a MultiIndex.
    result = df.groupby(["state_name", "county_name", "date_local"])["sample_measurement"]\
        .agg(["min", "max"])

    # Rename some columns and convert the MultiIndex of the 'result' DataFrame
    # into regular columns. This flattens the DataFrame so 'state_name',
    # 'county_name', and 'date_local' become regular columns again.
    result = result.rename(columns={'min': 'min_temperature', 'max': 'max_temperature'})\
        .reset_index()

    # Return the processed BigFrames DataFrame.
    # In a dbt Python model, this DataFrame will be materialized as a table
    return result
