SELECT
  regexp_contains(`t0`.`string_col`, '\\p{Lu}')
  AND NOT (
    regexp_contains(`t0`.`string_col`, '\\p{Ll}|\\p{Lt}')
  ) AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`