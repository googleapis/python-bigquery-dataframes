SELECT
  CASE
    WHEN `t1`.`int64_col` IS NULL
    THEN NULL
    WHEN TRUE
    THEN COALESCE(SUM(`t1`.`int64_col`) OVER (), 0)
    ELSE CAST(NULL AS INT64)
  END AS `agg_int64`
FROM (
  SELECT
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`