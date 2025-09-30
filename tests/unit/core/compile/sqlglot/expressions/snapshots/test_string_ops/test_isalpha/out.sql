SELECT
  regexp_contains(`t0`.`string_col`, '^(\\p{Lm}|\\p{Lt}|\\p{Lu}|\\p{Ll}|\\p{Lo})+$') AS `string_col`
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`