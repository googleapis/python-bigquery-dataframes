SELECT
  REGEXP_REPLACE(`t0`.`string_col`, 'e', 'a') AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`