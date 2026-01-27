SELECT
  `t0`.`bool_col`,
  `t0`.`float64_col` <> 0 AS `float64_col`,
  `t0`.`float64_col` <> 0 AS `float64_w_safe`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`