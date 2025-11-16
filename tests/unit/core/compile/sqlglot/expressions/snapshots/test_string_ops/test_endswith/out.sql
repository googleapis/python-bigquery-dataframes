SELECT
  ENDS_WITH(`t0`.`string_col`, 'ab') AS `single`,
  ENDS_WITH(`t0`.`string_col`, 'ab') OR ENDS_WITH(`t0`.`string_col`, 'cd') AS `double`,
  FALSE AS `empty`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`