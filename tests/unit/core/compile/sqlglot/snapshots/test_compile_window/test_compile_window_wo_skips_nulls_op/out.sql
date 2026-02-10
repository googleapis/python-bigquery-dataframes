SELECT
`rowindex` AS `rowindex`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t1`.`rowindex`,
  CASE
    WHEN COUNT((
      `t1`.`int64_col`
    ) IS NOT NULL) OVER (
      ORDER BY `t1`.`rowindex` IS NULL ASC, `t1`.`rowindex` ASC
      ROWS BETWEEN 4 preceding AND CURRENT ROW
    ) < 5
    THEN NULL
    WHEN TRUE
    THEN COUNT(`t1`.`int64_col`) OVER (
      ORDER BY `t1`.`rowindex` IS NULL ASC, `t1`.`rowindex` ASC
      ROWS BETWEEN 4 preceding AND CURRENT ROW
    )
    ELSE CAST(NULL AS INT64)
  END AS `int64_col`
FROM (
  SELECT
    `t0`.`int64_col`,
    `t0`.`rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`)
ORDER BY `rowindex` ASC NULLS LAST