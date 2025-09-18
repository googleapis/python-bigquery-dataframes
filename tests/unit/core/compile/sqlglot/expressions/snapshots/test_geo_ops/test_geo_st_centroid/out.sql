SELECT
  st_centroid(`t0`.`geography_col`) AS `geography_col`
FROM (
  SELECT
    `geography_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
) AS `t0`