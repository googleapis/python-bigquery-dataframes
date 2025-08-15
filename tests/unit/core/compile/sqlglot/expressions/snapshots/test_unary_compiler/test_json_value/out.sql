SELECT
  json_value(`t0`.`json_col`, '$') AS `json_col`
FROM (
  SELECT
    `json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.661510')
) AS `t0`