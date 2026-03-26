SELECT
  `t0`.`rowindex`,
  ROUND(`t0`.`int64_col` + `t0`.`int64_too`) AS `0`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`