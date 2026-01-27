SELECT
  COALESCE(LOGICAL_OR(`t1`.`bool_col`) OVER (), FALSE) AS `agg_bool`
FROM (
  SELECT
    `t0`.`bool_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`