SELECT
`bool_col` AS `bool_col`,
`rowindex` AS `rowindex`,
`bool_col_1` AS `bool_col_1`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t3`.`bfuid_col_392` AS `bool_col`,
  `t3`.`bfuid_col_389` AS `rowindex`,
  `t3`.`bfuid_col_393` AS `bool_col_1`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t3`.`bfuid_col_391`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t3`.`bfuid_col_392`
        ORDER BY `t3`.`bfuid_col_392` IS NULL ASC, `t3`.`bfuid_col_392` ASC, `t3`.`bfuid_col_398` IS NULL ASC, `t3`.`bfuid_col_398` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    ELSE COALESCE(
      SUM(`t3`.`bfuid_col_391`) OVER (
        PARTITION BY `t3`.`bfuid_col_392`
        ORDER BY `t3`.`bfuid_col_392` IS NULL ASC, `t3`.`bfuid_col_392` ASC, `t3`.`bfuid_col_398` IS NULL ASC, `t3`.`bfuid_col_398` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
  END AS `int64_col`,
  `t3`.`bfuid_col_398` AS `bfuid_col_399`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t1`.`bfuid_col_389`,
      `t1`.`bfuid_col_391`,
      `t1`.`bfuid_col_392`,
      CASE
        WHEN COALESCE(
          SUM(CAST((
            `t1`.`bfuid_col_390`
          ) IS NOT NULL AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_392`
            ORDER BY `t1`.`bfuid_col_392` IS NULL ASC, `t1`.`bfuid_col_392` ASC, `t1`.`bfuid_col_397` IS NULL ASC, `t1`.`bfuid_col_397` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        ) < 3
        THEN NULL
        ELSE COALESCE(
          SUM(CAST(`t1`.`bfuid_col_390` AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_392`
            ORDER BY `t1`.`bfuid_col_392` IS NULL ASC, `t1`.`bfuid_col_392` ASC, `t1`.`bfuid_col_397` IS NULL ASC, `t1`.`bfuid_col_397` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        )
      END AS `bfuid_col_393`,
      `t1`.`bfuid_col_397` AS `bfuid_col_398`
    FROM (
      SELECT
        `t0`.`rowindex` AS `bfuid_col_389`,
        `t0`.`bool_col` AS `bfuid_col_390`,
        `t0`.`int64_col` AS `bfuid_col_391`,
        `t0`.`bool_col` AS `bfuid_col_392`,
        `t0`.`rowindex` AS `bfuid_col_397`
      FROM (
        SELECT
          `bool_col`,
          `int64_col`,
          `rowindex`
        FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
      ) AS `t0`
      WHERE
        (
          `t0`.`bool_col`
        ) IS NOT NULL
    ) AS `t1`
  ) AS `t2`
  WHERE
    (
      `t2`.`bfuid_col_392`
    ) IS NOT NULL
) AS `t3`)
ORDER BY `bool_col` ASC NULLS LAST ,`bfuid_col_399` ASC NULLS LAST