SELECT
`rowindex` AS `rowindex`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`
FROM (
  SELECT
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`)
ORDER BY `int64_col` ASC NULLS LAST