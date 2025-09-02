SELECT
`bool_col` AS `bool_col`,
`int64_too` AS `int64_too`
FROM
(SELECT
  `t2`.`bfuid_col_758` AS `bool_col`,
  `t2`.`bfuid_col_759` AS `int64_too`
FROM (
  SELECT
    `t1`.`bfuid_col_758`,
    COALESCE(SUM(`t1`.`bfuid_col_757`), 0) AS `bfuid_col_759`
  FROM (
    SELECT
      `t0`.`int64_too` AS `bfuid_col_757`,
      `t0`.`bool_col` AS `bfuid_col_758`
    FROM (
      SELECT
        `bool_col`,
        `int64_too`
      FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
    ) AS `t0`
  ) AS `t1`
  GROUP BY
    1
) AS `t2`
WHERE
  (
    `t2`.`bfuid_col_758`
  ) IS NOT NULL)
ORDER BY `bool_col` ASC NULLS LAST