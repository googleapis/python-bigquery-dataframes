SELECT
  CAST(CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP) AS DATETIME) AS `int64_to_datetime`,
  TIME(CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP)) AS `int64_to_time`,
  CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP) AS `int64_to_timestamp`,
  TIME(CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP)) AS `int64_to_time_safe`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`