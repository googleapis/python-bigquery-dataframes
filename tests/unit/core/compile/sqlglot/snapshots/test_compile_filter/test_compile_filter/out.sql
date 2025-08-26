SELECT
  `t0`.`rowindex`,
  `t0`.`rowindex` AS `rowindex_1`,
  `t0`.`int64_col`
FROM (
  SELECT
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`
WHERE
  `t0`.`rowindex` >= 1