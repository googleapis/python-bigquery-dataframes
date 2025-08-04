# Implementing GeoSeries scalar operators

This project is to implement all GeoSeries scalar properties and methods in the
`bigframes.geopandas.GeoSeries` class. Likewise, all BigQuery GEOGRAPHY
functions should be exposed in the `bigframes.bigquery` module.

## Background

*Explain the context and why this change is necessary.*
*Include links to relevant issues or documentation.*

* https://geopandas.org/en/stable/docs/reference/geoseries.html
* https://cloud.google.com/bigquery/docs/reference/standard-sql/geography_functions

## Acceptance Criteria

*Define the specific, measurable outcomes that indicate the task is complete.*
*Use a checklist format for clarity.*

GeoSeries methods and properties

- [ ] Criterion 1
- [ ] Criterion 2

`bigframes.pandas` methods

- [ ] Criterion 3

## Detailed Steps

*Break down the implementation into small, actionable steps.*
*This section will guide the development process.*

### 1. Step One

- [ ] Action 1.1
- [ ] Action 1.2

### 2. Step Two

- [ ] Action 2.1
- [ ] Action 2.2

## Verification

*Specify the commands to run to verify the changes.*

- [ ] The `nox -r -s format lint lint_setup_py` linter should pass.
- [ ] The `nox -r -s mypy` static type checker should pass.
- [ ] The `nox -r -s docs docfx` docs should successfully build and include relevant docs in the output.
- [ ] All new and existing unit tests `pytest tests/unit` should pass.
- [ ] Identify all related system tests in the `tests/system` directories.
- [ ] All related system tests `pytest tests/system/small/path_to_relevant_test.py::test_name` should pass.

## Constraints

Follow the guidelines listed in GEMINI.md at the root of the repository.
