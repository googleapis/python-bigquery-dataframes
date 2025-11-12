SELECT
  `t0`.`rowindex`,
  `t0`.`bool_col`,
  `t0`.`int64_col`,
  `t0`.`int64_col` | `t0`.`int64_col` AS `int_and_int`,
  `t0`.`bool_col` OR `t0`.`bool_col` AS `bool_and_bool`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`