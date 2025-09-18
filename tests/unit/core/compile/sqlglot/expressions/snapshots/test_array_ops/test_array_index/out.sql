SELECT
  `t0`.`string_list_col`[safe_offset(1)] AS `string_list_col`
FROM (
  SELECT
    `string_list_col`
  FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.924678')
) AS `t0`