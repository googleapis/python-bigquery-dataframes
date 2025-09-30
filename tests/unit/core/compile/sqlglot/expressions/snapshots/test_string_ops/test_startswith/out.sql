SELECT
  STARTS_WITH(`t0`.`string_col`, 'ab') AS `single`,
  STARTS_WITH(`t0`.`string_col`, 'ab') OR STARTS_WITH(`t0`.`string_col`, 'cd') AS `double`,
  FALSE AS `empty`
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`