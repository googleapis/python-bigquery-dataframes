SELECT
  CEIL(`t1`.`float64_col`) AS `float64_col`
FROM (
  SELECT
    `t0`.`float64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`