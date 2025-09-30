SELECT
`bool_col` AS `bool_col`,
`rowindex` AS `rowindex`,
`bool_col_1` AS `bool_col_1`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t3`.`bfuid_col_909` AS `bool_col`,
  `t3`.`bfuid_col_906` AS `rowindex`,
  `t3`.`bfuid_col_910` AS `bool_col_1`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t3`.`bfuid_col_908`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t3`.`bfuid_col_909`
        ORDER BY `t3`.`bfuid_col_909` IS NULL ASC, `t3`.`bfuid_col_909` ASC, `t3`.`bfuid_col_915` IS NULL ASC, `t3`.`bfuid_col_915` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    WHEN TRUE
    THEN COALESCE(
      SUM(`t3`.`bfuid_col_908`) OVER (
        PARTITION BY `t3`.`bfuid_col_909`
        ORDER BY `t3`.`bfuid_col_909` IS NULL ASC, `t3`.`bfuid_col_909` ASC, `t3`.`bfuid_col_915` IS NULL ASC, `t3`.`bfuid_col_915` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
    ELSE CAST(NULL AS INT64)
  END AS `int64_col`,
  `t3`.`bfuid_col_915` AS `bfuid_col_916`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t1`.`bfuid_col_906`,
      `t1`.`bfuid_col_908`,
      `t1`.`bfuid_col_909`,
      CASE
        WHEN COALESCE(
          SUM(CAST((
            `t1`.`bfuid_col_907`
          ) IS NOT NULL AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_909`
            ORDER BY `t1`.`bfuid_col_909` IS NULL ASC, `t1`.`bfuid_col_909` ASC, `t1`.`bfuid_col_914` IS NULL ASC, `t1`.`bfuid_col_914` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        ) < 3
        THEN NULL
        WHEN TRUE
        THEN COALESCE(
          SUM(CAST(`t1`.`bfuid_col_907` AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_909`
            ORDER BY `t1`.`bfuid_col_909` IS NULL ASC, `t1`.`bfuid_col_909` ASC, `t1`.`bfuid_col_914` IS NULL ASC, `t1`.`bfuid_col_914` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        )
        ELSE CAST(NULL AS INT64)
      END AS `bfuid_col_910`,
      `t1`.`bfuid_col_914` AS `bfuid_col_915`
    FROM (
      SELECT
        `t0`.`rowindex` AS `bfuid_col_906`,
        `t0`.`bool_col` AS `bfuid_col_907`,
        `t0`.`int64_col` AS `bfuid_col_908`,
        `t0`.`bool_col` AS `bfuid_col_909`,
        `t0`.`rowindex` AS `bfuid_col_914`
      FROM (
        SELECT
          `bool_col`,
          `int64_col`,
          `rowindex`
        FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
      ) AS `t0`
      WHERE
        (
          `t0`.`bool_col`
        ) IS NOT NULL
    ) AS `t1`
  ) AS `t2`
  WHERE
    (
      `t2`.`bfuid_col_909`
    ) IS NOT NULL
) AS `t3`)
ORDER BY `bool_col` ASC NULLS LAST ,`bfuid_col_916` ASC NULLS LAST