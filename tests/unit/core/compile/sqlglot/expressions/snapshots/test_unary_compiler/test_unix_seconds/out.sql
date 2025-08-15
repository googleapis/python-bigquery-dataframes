SELECT
  UNIX_SECONDS(`t0`.`timestamp_col`) AS `timestamp_col`
FROM (
  SELECT
    `timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
) AS `t0`