SELECT
  CASE WHEN `t0`.`string_col` = 'value1' THEN 'mapped1' ELSE `t0`.`string_col` END AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`