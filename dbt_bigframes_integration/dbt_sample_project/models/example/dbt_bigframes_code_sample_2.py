# This example demonstrates how to build an incremental model.
#
# It applies lightweight, row-level logic to update or insert records into a
# target BigQuery table. If the target table already exists, dbt will perform a
# merge based on the specified unique keys; otherwise, it will create a new
# table automatically.
#
# It also defines and applies a BigFrames UDF to add a descriptive summary
# column based on temperature data.


import bigframes.pandas as bpd

def model(dbt, session):
    # Optional: override settings from dbt_project.yml.
    # When both are set, dbt.config takes precedence over dbt_project.yml.
    dbt.config(
        # Use BigFrames mode to execute the Python model.
        submission_method="bigframes",
        # Materialize as an incremental model.
        materialized='incremental',
        # Use MERGE strategy to update rows during incremental runs.
        incremental_strategy='merge',
        # Composite key to match existing rows for updates.
        unique_key=["state_name", "county_name", "date_local"],
    )

    # Reference an upstream dbt model or table as a DataFrame input.
    df = dbt.ref("dbt_bigframes_code_sample_1")

    # Define a BigFrames UDF to generate a temperature description.
    @bpd.udf(dataset='dbt_sample_dataset', name='describe_udf')
    def describe(
        max_temperature: float,
        min_temperature: float,
    ) -> str:
        is_hot = max_temperature > 85.0
        is_cold = min_temperature < 50.0

        if is_hot and is_cold:
            return "Expect both hot and cold conditions today."
        if is_hot:
            return "Overall, it's a hot day."
        if is_cold:
            return "Overall, it's a cold day."
        return "Comfortable throughout the day."

    # Apply the UDF using combine and store the result in a column "describe".
    df["describe"] = df["max_temperature"].combine(df["min_temperature"], describe)

    # Return the transformed DataFrame as the final dbt model output.
    return df