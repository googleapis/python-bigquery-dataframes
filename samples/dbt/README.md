# dbt BigFrames Integration

This repository provides simple examples of using **dbt Python models** with **BigQuery** in **BigFrames** mode.

It includes basic configurations and sample models to help you get started quickly in a typical dbt project.

## Highlights

- `profiles.yml`: configures your connection to BigQuery.
- `dbt_project.yml`: configures your dbt project - **dbt_sample_project**.
- `dbt_bigframes_code_sample_1.py`: An example to read BigQuery data and perform basic transformation.
- `dbt_bigframes_code_sample_2.py`: An example to build an incremental model that leverages BigFrames UDF capabilities.

## Requirements

Before using this project, ensure you have:

- A [Google Cloud account](https://cloud.google.com/free?hl=en)
- A [dbt Cloud account](https://www.getdbt.com/signup) (if using dbt Cloud)
- Python and SQL basics
- Familiarity with dbt concepts and structure

For more, see:
- https://docs.getdbt.com/guides/dbt-python-bigframes
- https://cloud.google.com/bigquery/docs/dataframes-dbt
