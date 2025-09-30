SELECT
  CAST(CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP) AS DATETIME) AS `int64_to_datetime`,
  TIME(CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP)) AS `int64_to_time`,
  CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP) AS `int64_to_timestamp`,
  TIME(CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP)) AS `int64_to_time_safe`
FROM (
  SELECT
    `int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`