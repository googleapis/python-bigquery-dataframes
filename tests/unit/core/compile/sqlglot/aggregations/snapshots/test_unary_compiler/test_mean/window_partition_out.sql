SELECT
  CASE
    WHEN `t1`.`int64_col` IS NULL
    THEN NULL
    WHEN TRUE
    THEN AVG(`t1`.`int64_col`) OVER (PARTITION BY `t1`.`string_col`)
    ELSE CAST(NULL AS FLOAT64)
  END AS `agg_int64`
FROM (
  SELECT
    `t0`.`int64_col`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`