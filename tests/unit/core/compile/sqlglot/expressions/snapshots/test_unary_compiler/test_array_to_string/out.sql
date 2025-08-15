SELECT
  ARRAY_TO_STRING(`t0`.`string_list_col`, '.') AS `string_list_col`
FROM (
  SELECT
    `string_list_col`
  FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:37.046629')
) AS `t0`