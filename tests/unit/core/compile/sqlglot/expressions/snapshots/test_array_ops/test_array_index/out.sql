SELECT
  IF(
    SUBSTRING(`t0`.`string_col`, IF((
      1 + 1
    ) >= 1, 1 + 1, 1 + 1 + LENGTH(`t0`.`string_col`)), 1) <> '',
    SUBSTRING(`t0`.`string_col`, IF((
      1 + 1
    ) >= 1, 1 + 1, 1 + 1 + LENGTH(`t0`.`string_col`)), 1),
    NULL
  ) AS `string_index`,
  [`t0`.`int64_col`, `t0`.`int64_too`][safe_offset(1)] AS `array_index`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`