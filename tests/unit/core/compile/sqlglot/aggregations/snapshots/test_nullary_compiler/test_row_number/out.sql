SELECT
  ROW_NUMBER() OVER (ORDER BY NULL ASC) - 1 AS `row_number`
FROM (
  SELECT
    `bool_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`