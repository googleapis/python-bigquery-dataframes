SELECT
`rowindex` AS `rowindex`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t0`.`rowindex`,
  CASE
    WHEN COUNT((
      `t0`.`int64_col`
    ) IS NOT NULL) OVER (
      ORDER BY `t0`.`rowindex` IS NULL ASC, `t0`.`rowindex` ASC
      ROWS BETWEEN 4 preceding AND CURRENT ROW
    ) < 5
    THEN NULL
    ELSE COUNT(`t0`.`int64_col`) OVER (
      ORDER BY `t0`.`rowindex` IS NULL ASC, `t0`.`rowindex` ASC
      ROWS BETWEEN 4 preceding AND CURRENT ROW
    )
  END AS `int64_col`
FROM (
  SELECT
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`)
ORDER BY `rowindex` ASC NULLS LAST