SELECT
  json_extract_string_array(`t0`.`json_col`, '$') AS `json_col`
FROM (
  SELECT
    `json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:50.715543')
) AS `t0`