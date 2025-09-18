SELECT
  CAST(EXTRACT(year FROM `t0`.`timestamp_col`) AS INT64) AS `timestamp_col`
FROM (
  SELECT
    `timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
) AS `t0`