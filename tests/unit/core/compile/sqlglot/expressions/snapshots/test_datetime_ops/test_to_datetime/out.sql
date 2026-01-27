SELECT
  CAST(CAST(timestamp_micros(CAST(trunc(`t0`.`int64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS DATETIME) AS `int64_col`,
  SAFE_CAST(`t0`.`string_col` AS DATETIME) AS `string_col`,
  CAST(CAST(timestamp_micros(CAST(trunc(`t0`.`float64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS DATETIME) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`