SELECT
  *
FROM (
  SELECT
    `rowindex`,
    `json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.661510')
) AS `t0`