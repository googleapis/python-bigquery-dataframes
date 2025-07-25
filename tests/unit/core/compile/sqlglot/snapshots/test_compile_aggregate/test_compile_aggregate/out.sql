SELECT
`bool_col` AS `bool_col`,
`int64_too` AS `int64_too`
FROM
(SELECT
  `t2`.`bfuid_col_59` AS `bool_col`,
  `t2`.`bfuid_col_60` AS `int64_too`
FROM (
  SELECT
    `t1`.`bfuid_col_59`,
    COALESCE(SUM(`t1`.`bfuid_col_58`), 0) AS `bfuid_col_60`
  FROM (
    SELECT
      `t0`.`int64_too` AS `bfuid_col_58`,
      `t0`.`bool_col` AS `bfuid_col_59`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
  GROUP BY
    1
) AS `t2`
WHERE
  (
    `t2`.`bfuid_col_59`
  ) IS NOT NULL)
ORDER BY `bool_col` ASC NULLS LAST