SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`bool_col`,
  `t0`.`int64_col` - `t0`.`int64_col` AS `int_sub_int`,
  `t0`.`int64_col` - 1 AS `int_sub_1`,
  CAST(NULL AS INT64) AS `int_sub_null`,
  `t0`.`int64_col` - CAST(`t0`.`bool_col` AS INT64) AS `int_sub_bool`,
  CAST(`t0`.`bool_col` AS INT64) - `t0`.`int64_col` AS `bool_sub_int`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`