SELECT
  DATETIME_DIFF(
    `t1`.`datetime_col`,
    LAG(`t1`.`datetime_col`, 1) OVER (ORDER BY `t1`.`datetime_col` IS NULL ASC, `t1`.`datetime_col` ASC),
    MICROSECOND
  ) AS `diff_datetime`
FROM (
  SELECT
    `t0`.`datetime_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`