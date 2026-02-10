SELECT
  IF(NOT (
    `t0`.`float64_col` >= 0
  ), CAST('NaN' AS FLOAT64), SQRT(`t0`.`float64_col`)) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`