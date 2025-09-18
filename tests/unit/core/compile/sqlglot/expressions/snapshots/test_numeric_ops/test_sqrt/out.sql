SELECT
  IF(NOT (
    `t0`.`float64_col` >= 0
  ), CAST('NaN' AS FLOAT64), SQRT(`t0`.`float64_col`)) AS `float64_col`
FROM (
  SELECT
    `float64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
) AS `t0`