SELECT
`rowindex` AS `rowindex`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t0`.`rowindex`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t0`.`int64_col`
      ) IS NOT NULL AS INT64)) OVER (
        ORDER BY `t0`.`rowindex` IS NULL ASC, `t0`.`rowindex` ASC
        ROWS BETWEEN 2 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    WHEN TRUE
    THEN COALESCE(
      SUM(`t0`.`int64_col`) OVER (
        ORDER BY `t0`.`rowindex` IS NULL ASC, `t0`.`rowindex` ASC
        ROWS BETWEEN 2 preceding AND CURRENT ROW
      ),
      0
    )
    ELSE CAST(NULL AS INT64)
  END AS `int64_col`
FROM (
  SELECT
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`)
ORDER BY `rowindex` ASC NULLS LAST