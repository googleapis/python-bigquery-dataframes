SELECT
  json_extract_string_array(`t0`.`json_col`, '$') AS `json_col`
FROM (
  SELECT
    `json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:47.763579')
) AS `t0`