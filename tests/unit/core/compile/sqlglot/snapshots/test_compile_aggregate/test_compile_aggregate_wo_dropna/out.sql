SELECT
`bool_col` AS `bool_col`,
`int64_too` AS `int64_too`
FROM
(SELECT
  `t2`.`bfuid_col_1146` AS `bool_col`,
  `t2`.`bfuid_col_1147` AS `int64_too`
FROM (
  SELECT
    `t1`.`bfuid_col_1146`,
    COALESCE(SUM(`t1`.`bfuid_col_1145`), 0) AS `bfuid_col_1147`
  FROM (
    SELECT
      `t0`.`int64_too` AS `bfuid_col_1145`,
      `t0`.`bool_col` AS `bfuid_col_1146`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
  GROUP BY
    1
) AS `t2`)
ORDER BY `bool_col` ASC NULLS LAST