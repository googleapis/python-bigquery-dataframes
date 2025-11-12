SELECT
  TIMESTAMP_DIFF(
    `t1`.`timestamp_col`,
    LAG(`t1`.`timestamp_col`, 1) OVER (ORDER BY `t1`.`timestamp_col` IS NULL ASC, `t1`.`timestamp_col` ASC),
    MICROSECOND
  ) AS `diff_time`
FROM (
  SELECT
    `t0`.`timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`