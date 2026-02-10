SELECT
  regexp_contains(`t0`.`string_col`, '\\p{Ll}')
  AND NOT (
    regexp_contains(`t0`.`string_col`, '\\p{Lu}|\\p{Lt}')
  ) AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`