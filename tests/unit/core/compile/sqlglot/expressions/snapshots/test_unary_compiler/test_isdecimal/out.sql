SELECT
  regexp_contains(`t0`.`string_col`, '^(\\p{Nd})+$') AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`