SELECT
`bool_col` AS `bool_col`,
`int64_too` AS `int64_too`
FROM
(SELECT
  `t2`.`bfuid_col_848` AS `bool_col`,
  `t2`.`bfuid_col_849` AS `int64_too`
FROM (
  SELECT
    `t1`.`bfuid_col_848`,
    COALESCE(SUM(`t1`.`bfuid_col_847`), 0) AS `bfuid_col_849`
  FROM (
    SELECT
      `t0`.`int64_too` AS `bfuid_col_847`,
      `t0`.`bool_col` AS `bfuid_col_848`
    FROM (
      SELECT
        `bool_col`,
        `int64_too`
      FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
    ) AS `t0`
  ) AS `t1`
  GROUP BY
    1
) AS `t2`)
ORDER BY `bool_col` ASC NULLS LAST