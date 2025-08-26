SELECT
  COALESCE(`t0`.`int64_col` IN (1, 2, 3), FALSE) AS `int64_col`
FROM (
  SELECT
    `int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`