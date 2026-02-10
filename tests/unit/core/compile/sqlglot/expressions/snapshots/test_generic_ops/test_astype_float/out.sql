SELECT
  CAST(CAST(`t0`.`bool_col` AS INT64) AS FLOAT64) AS `bool_col`,
  CAST('1.34235e4' AS FLOAT64) AS `str_const`,
  SAFE_CAST(SAFE_CAST(`t0`.`bool_col` AS INT64) AS FLOAT64) AS `bool_w_safe`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`