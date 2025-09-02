SELECT
  st_buffer(`t0`.`geography_col`, 1.0, 8.0, FALSE) AS `geography_col`
FROM (
  SELECT
    `geography_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`