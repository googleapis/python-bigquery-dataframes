SELECT
  CASE WHEN `t0`.`string_col` = 'value1' THEN 'mapped1' ELSE `t0`.`string_col` END AS `string_col`
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`