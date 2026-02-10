SELECT
`rowindex` AS `rowindex`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`)
ORDER BY `rowindex` ASC NULLS LAST
LIMIT 10