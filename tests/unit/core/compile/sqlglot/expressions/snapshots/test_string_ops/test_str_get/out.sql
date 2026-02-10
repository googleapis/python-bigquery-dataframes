SELECT
  NULLIF(
    SUBSTRING(`t0`.`string_col`, IF((
      1 + 1
    ) >= 1, 1 + 1, 1 + 1 + LENGTH(`t0`.`string_col`)), 1),
    ''
  ) AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`