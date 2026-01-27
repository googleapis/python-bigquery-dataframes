SELECT
`bool_col` AS `bool_col`,
`rowindex` AS `rowindex`,
`bool_col_1` AS `bool_col_1`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t1`.`bfuid_col_1518` AS `bool_col`,
  `t1`.`bfuid_col_1515` AS `rowindex`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t1`.`bfuid_col_1516`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t1`.`bfuid_col_1518`
        ORDER BY `t1`.`bfuid_col_1518` IS NULL ASC, `t1`.`bfuid_col_1518` ASC, `t1`.`bfuid_col_1523` IS NULL ASC, `t1`.`bfuid_col_1523` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    WHEN TRUE
    THEN COALESCE(
      SUM(CAST(`t1`.`bfuid_col_1516` AS INT64)) OVER (
        PARTITION BY `t1`.`bfuid_col_1518`
        ORDER BY `t1`.`bfuid_col_1518` IS NULL ASC, `t1`.`bfuid_col_1518` ASC, `t1`.`bfuid_col_1523` IS NULL ASC, `t1`.`bfuid_col_1523` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
    ELSE CAST(NULL AS INT64)
  END AS `bool_col_1`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t1`.`bfuid_col_1517`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t1`.`bfuid_col_1518`
        ORDER BY `t1`.`bfuid_col_1518` IS NULL ASC, `t1`.`bfuid_col_1518` ASC, `t1`.`bfuid_col_1523` IS NULL ASC, `t1`.`bfuid_col_1523` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    WHEN TRUE
    THEN COALESCE(
      SUM(`t1`.`bfuid_col_1517`) OVER (
        PARTITION BY `t1`.`bfuid_col_1518`
        ORDER BY `t1`.`bfuid_col_1518` IS NULL ASC, `t1`.`bfuid_col_1518` ASC, `t1`.`bfuid_col_1523` IS NULL ASC, `t1`.`bfuid_col_1523` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
    ELSE CAST(NULL AS INT64)
  END AS `int64_col`,
  `t1`.`bfuid_col_1523` AS `bfuid_col_1524`
FROM (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1515`,
    `t0`.`bool_col` AS `bfuid_col_1516`,
    `t0`.`int64_col` AS `bfuid_col_1517`,
    `t0`.`bool_col` AS `bfuid_col_1518`,
    `t0`.`rowindex` AS `bfuid_col_1523`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  WHERE
    (
      `t0`.`bool_col`
    ) IS NOT NULL
) AS `t1`)
ORDER BY `bool_col` ASC NULLS LAST ,`bfuid_col_1524` ASC NULLS LAST