SELECT
  MIN(`t1`.`int64_col`) OVER () AS `agg_int64`
FROM (
  SELECT
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`