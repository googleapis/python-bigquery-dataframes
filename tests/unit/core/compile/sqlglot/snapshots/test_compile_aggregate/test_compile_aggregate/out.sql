SELECT `bool_col`, `int64_too` FROM (SELECT
  `t2`.`bfuid_col_1532` AS `bool_col`,
  `t2`.`bfuid_col_1533` AS `int64_too`
FROM (
  SELECT
    `t1`.`bfuid_col_1532`,
    COALESCE(SUM(`t1`.`bfuid_col_1531`), 0) AS `bfuid_col_1533`
  FROM (
    SELECT
      `t0`.`int64_too` AS `bfuid_col_1531`,
      `t0`.`bool_col` AS `bfuid_col_1532`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
  GROUP BY
    1
) AS `t2`
WHERE
  (
    `t2`.`bfuid_col_1532`
  ) IS NOT NULL) AS `t`
ORDER BY `bool_col` ASC NULLS LAST