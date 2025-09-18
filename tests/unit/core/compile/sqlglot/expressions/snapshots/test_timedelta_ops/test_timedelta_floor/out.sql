SELECT
  CAST(FLOOR(`t0`.`int64_col`) AS INT64) AS `int64_col`
FROM (
  SELECT
    `int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
) AS `t0`