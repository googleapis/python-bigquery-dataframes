SELECT
  unix_micros(CAST(`t0`.`datetime_col` AS TIMESTAMP)) AS `datetime_col`,
  SAFE_CAST(SAFE_CAST(`t0`.`datetime_col` AS TIMESTAMP) AS INT64) AS `datetime_w_safe`,
  TIME_DIFF(`t0`.`time_col`, TIME(0, 0, 0), MICROSECOND) AS `time_col`,
  TIME_DIFF(`t0`.`time_col`, TIME(0, 0, 0), MICROSECOND) AS `time_w_safe`,
  unix_micros(`t0`.`timestamp_col`) AS `timestamp_col`,
  CAST(trunc(`t0`.`numeric_col`) AS INT64) AS `numeric_col`,
  CAST(trunc(`t0`.`float64_col`) AS INT64) AS `float64_col`,
  SAFE_CAST(`t0`.`float64_col` AS INT64) AS `float64_w_safe`,
  CAST('100' AS INT64) AS `str_const`
FROM (
  SELECT
    `datetime_col`,
    `numeric_col`,
    `float64_col`,
    `time_col`,
    `timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`