SELECT
  CAST(timestamp_micros(CAST(trunc(`t0`.`int64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS `int64_col`,
  CAST(timestamp_micros(CAST(trunc(`t0`.`float64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS `float64_col`,
  CAST(timestamp_micros(`t0`.`int64_col` * 1000000) AS TIMESTAMP) AS `int64_col_s`,
  CAST(timestamp_micros(`t0`.`int64_col` * 1000) AS TIMESTAMP) AS `int64_col_ms`,
  CAST(timestamp_micros(`t0`.`int64_col` * 1) AS TIMESTAMP) AS `int64_col_us`,
  CAST(timestamp_micros(CAST(trunc(`t0`.`int64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS `int64_col_ns`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`