SELECT `rowindex`, `int64_col` FROM (SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`) AS `t`
ORDER BY `int64_col` ASC NULLS LAST