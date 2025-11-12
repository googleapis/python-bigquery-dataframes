SELECT
`bool_col` AS `bool_col`,
`rowindex` AS `rowindex`,
`bool_col_1` AS `bool_col_1`,
`int64_col` AS `int64_col`
FROM
(SELECT
  `t3`.`bfuid_col_1290` AS `bool_col`,
  `t3`.`bfuid_col_1287` AS `rowindex`,
  `t3`.`bfuid_col_1291` AS `bool_col_1`,
  CASE
    WHEN COALESCE(
      SUM(CAST((
        `t3`.`bfuid_col_1289`
      ) IS NOT NULL AS INT64)) OVER (
        PARTITION BY `t3`.`bfuid_col_1290`
        ORDER BY `t3`.`bfuid_col_1290` IS NULL ASC, `t3`.`bfuid_col_1290` ASC, `t3`.`bfuid_col_1296` IS NULL ASC, `t3`.`bfuid_col_1296` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    ) < 3
    THEN NULL
    WHEN TRUE
    THEN COALESCE(
      SUM(`t3`.`bfuid_col_1289`) OVER (
        PARTITION BY `t3`.`bfuid_col_1290`
        ORDER BY `t3`.`bfuid_col_1290` IS NULL ASC, `t3`.`bfuid_col_1290` ASC, `t3`.`bfuid_col_1296` IS NULL ASC, `t3`.`bfuid_col_1296` ASC
        ROWS BETWEEN 3 preceding AND CURRENT ROW
      ),
      0
    )
    ELSE CAST(NULL AS INT64)
  END AS `int64_col`,
  `t3`.`bfuid_col_1296` AS `bfuid_col_1297`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t1`.`bfuid_col_1287`,
      `t1`.`bfuid_col_1289`,
      `t1`.`bfuid_col_1290`,
      CASE
        WHEN COALESCE(
          SUM(CAST((
            `t1`.`bfuid_col_1288`
          ) IS NOT NULL AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_1290`
            ORDER BY `t1`.`bfuid_col_1290` IS NULL ASC, `t1`.`bfuid_col_1290` ASC, `t1`.`bfuid_col_1295` IS NULL ASC, `t1`.`bfuid_col_1295` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        ) < 3
        THEN NULL
        WHEN TRUE
        THEN COALESCE(
          SUM(CAST(`t1`.`bfuid_col_1288` AS INT64)) OVER (
            PARTITION BY `t1`.`bfuid_col_1290`
            ORDER BY `t1`.`bfuid_col_1290` IS NULL ASC, `t1`.`bfuid_col_1290` ASC, `t1`.`bfuid_col_1295` IS NULL ASC, `t1`.`bfuid_col_1295` ASC
            ROWS BETWEEN 3 preceding AND CURRENT ROW
          ),
          0
        )
        ELSE CAST(NULL AS INT64)
      END AS `bfuid_col_1291`,
      `t1`.`bfuid_col_1295` AS `bfuid_col_1296`
    FROM (
      SELECT
        `t0`.`rowindex` AS `bfuid_col_1287`,
        `t0`.`bool_col` AS `bfuid_col_1288`,
        `t0`.`int64_col` AS `bfuid_col_1289`,
        `t0`.`bool_col` AS `bfuid_col_1290`,
        `t0`.`rowindex` AS `bfuid_col_1295`
      FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
      WHERE
        (
          `t0`.`bool_col`
        ) IS NOT NULL
    ) AS `t1`
  ) AS `t2`
  WHERE
    (
      `t2`.`bfuid_col_1290`
    ) IS NOT NULL
) AS `t3`)
ORDER BY `bool_col` ASC NULLS LAST ,`bfuid_col_1297` ASC NULLS LAST