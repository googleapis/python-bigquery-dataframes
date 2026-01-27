SELECT
  `t0`.`rowindex`,
  `t0`.`bool_col`,
  `t0`.`int64_col`,
  `t0`.`int64_col` & `t0`.`int64_col` AS `int_and_int`,
  `t0`.`bool_col` AND `t0`.`bool_col` AS `bool_and_bool`,
  IF(`t0`.`bool_col` = FALSE, `t0`.`bool_col`, NULL) AS `bool_and_null`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`