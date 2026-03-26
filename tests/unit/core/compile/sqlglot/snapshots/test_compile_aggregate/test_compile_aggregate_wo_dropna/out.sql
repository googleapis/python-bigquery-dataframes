SELECT `bool_col`, `int64_too` FROM (SELECT
  `t2`.`bfuid_col_1536` AS `bool_col`,
  `t2`.`bfuid_col_1537` AS `int64_too`
FROM (
  SELECT
    `t1`.`bfuid_col_1536`,
    COALESCE(SUM(`t1`.`bfuid_col_1535`), 0) AS `bfuid_col_1537`
  FROM (
    SELECT
      `t0`.`int64_too` AS `bfuid_col_1535`,
      `t0`.`bool_col` AS `bfuid_col_1536`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
  GROUP BY
    1
) AS `t2`) AS `t`
ORDER BY `bool_col` ASC NULLS LAST