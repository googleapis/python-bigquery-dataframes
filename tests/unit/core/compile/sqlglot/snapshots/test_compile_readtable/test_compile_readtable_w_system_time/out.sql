WITH `bfcte_0` AS (
  SELECT
    `bool_col`,
    `bytes_col`,
    `date_col`,
    `datetime_col`,
    `duration_col`,
    `float64_col`,
    `geography_col`,
    `int64_col`,
    `int64_too`,
    `numeric_col`,
    `rowindex`,
    `rowindex_2`,
    `string_col`,
    `time_col`,
    `timestamp_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` FOR SYSTEM_TIME AS OF '2025-11-09T03:04:05.678901+00:00'
)
SELECT
  *,
  `bool_col` AS `bool_col`,
  `bytes_col` AS `bytes_col`,
  `date_col` AS `date_col`,
  `datetime_col` AS `datetime_col`,
  `geography_col` AS `geography_col`,
  `int64_col` AS `int64_col`,
  `int64_too` AS `int64_too`,
  `numeric_col` AS `numeric_col`,
  `float64_col` AS `float64_col`,
  `rowindex` AS `rowindex`,
  `rowindex_2` AS `rowindex_2`,
  `string_col` AS `string_col`,
  `time_col` AS `time_col`,
  `timestamp_col` AS `timestamp_col`,
  `duration_col` AS `duration_col`
FROM `bfcte_0`