SELECT
  st_area(`t0`.`geography_col`) AS `geography_col`
FROM (
  SELECT
    `geography_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
) AS `t0`