SELECT
  STRUCT(
    `t0`.`bool_col` AS `bool_col`,
    `t0`.`int64_col` AS `int64_col`,
    `t0`.`float64_col` AS `float64_col`,
    `t0`.`string_col` AS `string_col`
  ) AS `result_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`