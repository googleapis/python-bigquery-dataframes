SELECT
  *
FROM (
  SELECT
    `rowindex`,
    `json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.866123')
) AS `t0`