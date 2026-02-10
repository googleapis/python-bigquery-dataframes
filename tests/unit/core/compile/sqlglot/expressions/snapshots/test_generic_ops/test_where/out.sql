SELECT
  CASE WHEN `t0`.`bool_col` THEN `t0`.`int64_col` ELSE `t0`.`float64_col` END AS `result_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`