SELECT
  UNIX_MICROS(CAST(`t0`.`datetime_col` AS TIMESTAMP)) AS `datetime_col`,
  SAFE_CAST(SAFE_CAST(`t0`.`datetime_col` AS TIMESTAMP) AS INT64) AS `datetime_w_safe`,
  TIME_DIFF(`t0`.`time_col`, TIME(0, 0, 0), MICROSECOND) AS `time_col`,
  TIME_DIFF(`t0`.`time_col`, TIME(0, 0, 0), MICROSECOND) AS `time_w_safe`,
  UNIX_MICROS(`t0`.`timestamp_col`) AS `timestamp_col`,
  CAST(trunc(`t0`.`numeric_col`) AS INT64) AS `numeric_col`,
  CAST(trunc(`t0`.`float64_col`) AS INT64) AS `float64_col`,
  SAFE_CAST(`t0`.`float64_col` AS INT64) AS `float64_w_safe`,
  CAST('100' AS INT64) AS `str_const`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`