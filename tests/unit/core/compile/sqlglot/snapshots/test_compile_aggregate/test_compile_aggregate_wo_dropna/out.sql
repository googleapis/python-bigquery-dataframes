SELECT
`bool_col` AS `bool_col`,
`int64_too` AS `int64_too`
FROM
(SELECT
  `t2`.`bfuid_col_1372` AS `bool_col`,
  `t2`.`bfuid_col_1373` AS `int64_too`
FROM (
  SELECT
    `t1`.`bfuid_col_1372`,
    COALESCE(SUM(`t1`.`bfuid_col_1371`), 0) AS `bfuid_col_1373`
  FROM (
    SELECT
      `t0`.`int64_too` AS `bfuid_col_1371`,
      `t0`.`bool_col` AS `bfuid_col_1372`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
  GROUP BY
    1
) AS `t2`)
ORDER BY `bool_col` ASC NULLS LAST