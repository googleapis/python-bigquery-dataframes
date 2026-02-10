SELECT
  COUNT(`t1`.`int64_col`) OVER (
    PARTITION BY `t1`.`string_col`
    ORDER BY `t1`.`int64_col` DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS `agg_int64`
FROM (
  SELECT
    `t0`.`int64_col`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`