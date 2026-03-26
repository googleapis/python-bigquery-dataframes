SELECT `bool_col`, `rowindex`, `bool_col_1`, `int64_col` FROM (SELECT
  `t1`.`bfuid_col_1690` AS `bool_col`,
  `t1`.`bfuid_col_1687` AS `rowindex`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t1`.`bfuid_col_1688`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t1`.`bfuid_col_1690`
        ORDER BY `t1`.`bfuid_col_1690` IS NULL ASC, `t1`.`bfuid_col_1690` ASC, `t1`.`bfuid_col_1695` IS NULL ASC, `t1`.`bfuid_col_1695` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    WHEN TRUE
    THEN COALESCE(
      SUM(CAST(`t1`.`bfuid_col_1688` AS INT64)) OVER (
        PARTITION BY `t1`.`bfuid_col_1690`
        ORDER BY `t1`.`bfuid_col_1690` IS NULL ASC, `t1`.`bfuid_col_1690` ASC, `t1`.`bfuid_col_1695` IS NULL ASC, `t1`.`bfuid_col_1695` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
    ELSE CAST(NULL AS INT64)
  END AS `bool_col_1`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t1`.`bfuid_col_1689`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t1`.`bfuid_col_1690`
        ORDER BY `t1`.`bfuid_col_1690` IS NULL ASC, `t1`.`bfuid_col_1690` ASC, `t1`.`bfuid_col_1695` IS NULL ASC, `t1`.`bfuid_col_1695` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    WHEN TRUE
    THEN COALESCE(
      SUM(`t1`.`bfuid_col_1689`) OVER (
        PARTITION BY `t1`.`bfuid_col_1690`
        ORDER BY `t1`.`bfuid_col_1690` IS NULL ASC, `t1`.`bfuid_col_1690` ASC, `t1`.`bfuid_col_1695` IS NULL ASC, `t1`.`bfuid_col_1695` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
    ELSE CAST(NULL AS INT64)
  END AS `int64_col`,
  `t1`.`bfuid_col_1695` AS `bfuid_col_1696`
FROM (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1687`,
    `t0`.`bool_col` AS `bfuid_col_1688`,
    `t0`.`int64_col` AS `bfuid_col_1689`,
    `t0`.`bool_col` AS `bfuid_col_1690`,
    `t0`.`rowindex` AS `bfuid_col_1695`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  WHERE
    (
      `t0`.`bool_col`
    ) IS NOT NULL
) AS `t1`) AS `t`
ORDER BY `bool_col` ASC NULLS LAST ,`bfuid_col_1696` ASC NULLS LAST