SELECT
`bool_col` AS `bool_col`,
`rowindex` AS `rowindex`,
`bool_col_1` AS `bool_col_1`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t3`.`bfuid_col_840` AS `bool_col`,
  `t3`.`bfuid_col_837` AS `rowindex`,
  `t3`.`bfuid_col_841` AS `bool_col_1`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t3`.`bfuid_col_839`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t3`.`bfuid_col_840`
        ORDER BY `t3`.`bfuid_col_840` IS NULL ASC, `t3`.`bfuid_col_840` ASC, `t3`.`bfuid_col_846` IS NULL ASC, `t3`.`bfuid_col_846` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    WHEN TRUE
    THEN COALESCE(
      SUM(`t3`.`bfuid_col_839`) OVER (
        PARTITION BY `t3`.`bfuid_col_840`
        ORDER BY `t3`.`bfuid_col_840` IS NULL ASC, `t3`.`bfuid_col_840` ASC, `t3`.`bfuid_col_846` IS NULL ASC, `t3`.`bfuid_col_846` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
    ELSE CAST(NULL AS INT64)
  END AS `int64_col`,
  `t3`.`bfuid_col_846` AS `bfuid_col_847`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t1`.`bfuid_col_837`,
      `t1`.`bfuid_col_839`,
      `t1`.`bfuid_col_840`,
      CASE
        WHEN COALESCE(
          SUM(CAST((
            `t1`.`bfuid_col_838`
          ) IS NOT NULL AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_840`
            ORDER BY `t1`.`bfuid_col_840` IS NULL ASC, `t1`.`bfuid_col_840` ASC, `t1`.`bfuid_col_845` IS NULL ASC, `t1`.`bfuid_col_845` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        ) < 3
        THEN NULL
        WHEN TRUE
        THEN COALESCE(
          SUM(CAST(`t1`.`bfuid_col_838` AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_840`
            ORDER BY `t1`.`bfuid_col_840` IS NULL ASC, `t1`.`bfuid_col_840` ASC, `t1`.`bfuid_col_845` IS NULL ASC, `t1`.`bfuid_col_845` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        )
        ELSE CAST(NULL AS INT64)
      END AS `bfuid_col_841`,
      `t1`.`bfuid_col_845` AS `bfuid_col_846`
    FROM (
      SELECT
        `t0`.`rowindex` AS `bfuid_col_837`,
        `t0`.`bool_col` AS `bfuid_col_838`,
        `t0`.`int64_col` AS `bfuid_col_839`,
        `t0`.`bool_col` AS `bfuid_col_840`,
        `t0`.`rowindex` AS `bfuid_col_845`
      FROM (
        SELECT
          `bool_col`,
          `int64_col`,
          `rowindex`
        FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
      ) AS `t0`
      WHERE
        (
          `t0`.`bool_col`
        ) IS NOT NULL
    ) AS `t1`
  ) AS `t2`
  WHERE
    (
      `t2`.`bfuid_col_840`
    ) IS NOT NULL
) AS `t3`)
ORDER BY `bool_col` ASC NULLS LAST ,`bfuid_col_847` ASC NULLS LAST