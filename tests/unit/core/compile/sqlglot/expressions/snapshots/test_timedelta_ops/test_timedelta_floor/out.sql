SELECT
  CASE
    WHEN `t0`.`int64_col` > 0
    THEN CAST(FLOOR(`t0`.`int64_col`) AS INT64)
    ELSE CAST(CEIL(`t0`.`int64_col`) AS INT64)
  END AS `int64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`