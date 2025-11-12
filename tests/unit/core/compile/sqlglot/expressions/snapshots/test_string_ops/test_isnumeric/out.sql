SELECT
  regexp_contains(`t0`.`string_col`, '^(\\pN+)$') AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`