SELECT
  regexp_contains(`t0`.`string_col`, '^(\\p{N}|\\p{Lm}|\\p{Lt}|\\p{Lu}|\\p{Ll}|\\p{Lo})+$') AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`