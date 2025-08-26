SELECT
`bool_col` AS `bool_col`,
`rowindex` AS `rowindex`,
`bool_col_1` AS `bool_col_1`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t3`.`bfuid_col_821` AS `bool_col`,
  `t3`.`bfuid_col_818` AS `rowindex`,
  `t3`.`bfuid_col_822` AS `bool_col_1`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t3`.`bfuid_col_820`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t3`.`bfuid_col_821`
        ORDER BY `t3`.`bfuid_col_821` IS NULL ASC, `t3`.`bfuid_col_821` ASC, `t3`.`bfuid_col_827` IS NULL ASC, `t3`.`bfuid_col_827` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    ELSE COALESCE(
      SUM(`t3`.`bfuid_col_820`) OVER (
        PARTITION BY `t3`.`bfuid_col_821`
        ORDER BY `t3`.`bfuid_col_821` IS NULL ASC, `t3`.`bfuid_col_821` ASC, `t3`.`bfuid_col_827` IS NULL ASC, `t3`.`bfuid_col_827` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
  END AS `int64_col`,
  `t3`.`bfuid_col_827` AS `bfuid_col_828`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t1`.`bfuid_col_818`,
      `t1`.`bfuid_col_820`,
      `t1`.`bfuid_col_821`,
      CASE
        WHEN COALESCE(
          SUM(CAST((
            `t1`.`bfuid_col_819`
          ) IS NOT NULL AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_821`
            ORDER BY `t1`.`bfuid_col_821` IS NULL ASC, `t1`.`bfuid_col_821` ASC, `t1`.`bfuid_col_826` IS NULL ASC, `t1`.`bfuid_col_826` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        ) < 3
        THEN NULL
        ELSE COALESCE(
          SUM(CAST(`t1`.`bfuid_col_819` AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_821`
            ORDER BY `t1`.`bfuid_col_821` IS NULL ASC, `t1`.`bfuid_col_821` ASC, `t1`.`bfuid_col_826` IS NULL ASC, `t1`.`bfuid_col_826` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        )
      END AS `bfuid_col_822`,
      `t1`.`bfuid_col_826` AS `bfuid_col_827`
    FROM (
      SELECT
        `t0`.`rowindex` AS `bfuid_col_818`,
        `t0`.`bool_col` AS `bfuid_col_819`,
        `t0`.`int64_col` AS `bfuid_col_820`,
        `t0`.`bool_col` AS `bfuid_col_821`,
        `t0`.`rowindex` AS `bfuid_col_826`
      FROM (
        SELECT
          `bool_col`,
          `int64_col`,
          `rowindex`
        FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
      ) AS `t0`
      WHERE
        (
          `t0`.`bool_col`
        ) IS NOT NULL
    ) AS `t1`
  ) AS `t2`
  WHERE
    (
      `t2`.`bfuid_col_821`
    ) IS NOT NULL
) AS `t3`)
ORDER BY `bool_col` ASC NULLS LAST ,`bfuid_col_828` ASC NULLS LAST