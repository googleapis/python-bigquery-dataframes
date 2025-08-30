SELECT
  st_astext(`t0`.`geography_col`) AS `geography_col`
FROM (
  SELECT
    `geography_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`