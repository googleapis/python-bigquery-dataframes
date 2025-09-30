SELECT
  atan(`t0`.`float64_col`) AS `float64_col`
FROM (
  SELECT
    `float64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`