SELECT
`bool_col` AS `bool_col`,
`int64_too` AS `int64_too`
FROM
(SELECT
  `t2`.`bfuid_col_330` AS `bool_col`,
  `t2`.`bfuid_col_331` AS `int64_too`
FROM (
  SELECT
    `t1`.`bfuid_col_330`,
    COALESCE(SUM(`t1`.`bfuid_col_329`), 0) AS `bfuid_col_331`
  FROM (
    SELECT
      `t0`.`int64_too` AS `bfuid_col_329`,
      `t0`.`bool_col` AS `bfuid_col_330`
    FROM (
      SELECT
        `bool_col`,
        `int64_too`
      FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
    ) AS `t0`
  ) AS `t1`
  GROUP BY
    1
) AS `t2`
WHERE
  (
    `t2`.`bfuid_col_330`
  ) IS NOT NULL)
ORDER BY `bool_col` ASC NULLS LAST