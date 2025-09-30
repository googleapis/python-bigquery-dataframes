SELECT
  CAST(CAST(`t0`.`bool_col` AS INT64) AS FLOAT64) AS `bool_col`,
  CAST('1.34235e4' AS FLOAT64) AS `str_const`,
  SAFE_CAST(SAFE_CAST(`t0`.`bool_col` AS INT64) AS FLOAT64) AS `bool_w_safe`
FROM (
  SELECT
    `bool_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`