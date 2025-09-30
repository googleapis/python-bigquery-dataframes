SELECT
  regexp_contains(`t0`.`string_col`, '\\p{Lu}')
  AND NOT (
    regexp_contains(`t0`.`string_col`, '\\p{Ll}|\\p{Lt}')
  ) AS `string_col`
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`