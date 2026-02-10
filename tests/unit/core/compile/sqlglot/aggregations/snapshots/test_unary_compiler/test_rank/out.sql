SELECT
  (
    RANK() OVER (ORDER BY `t1`.`int64_col` IS NULL DESC, `t1`.`int64_col` DESC) - 1
  ) + 1 AS `agg_int64`
FROM (
  SELECT
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`