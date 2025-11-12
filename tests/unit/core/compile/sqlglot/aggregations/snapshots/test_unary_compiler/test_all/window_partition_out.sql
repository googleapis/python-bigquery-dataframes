SELECT
  CASE
    WHEN `t1`.`bool_col` IS NULL
    THEN NULL
    WHEN TRUE
    THEN COALESCE(LOGICAL_AND(`t1`.`bool_col`) OVER (PARTITION BY `t1`.`string_col`), TRUE)
    ELSE CAST(NULL AS BOOL)
  END AS `agg_bool`
FROM (
  SELECT
    `t0`.`bool_col`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`