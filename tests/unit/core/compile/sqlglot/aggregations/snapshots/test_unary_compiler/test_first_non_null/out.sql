SELECT
  FIRST_VALUE(`t1`.`int64_col` IGNORE NULLS) OVER (
    ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS `agg_int64`
FROM (
  SELECT
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`