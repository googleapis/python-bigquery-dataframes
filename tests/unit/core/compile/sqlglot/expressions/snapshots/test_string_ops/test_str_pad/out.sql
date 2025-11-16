SELECT
  LPAD(`t0`.`string_col`, GREATEST(LENGTH(`t0`.`string_col`), 10), '-') AS `left`,
  RPAD(`t0`.`string_col`, GREATEST(LENGTH(`t0`.`string_col`), 10), '-') AS `right`,
  RPAD(
    LPAD(
      `t0`.`string_col`,
      (
        CAST(FLOOR(
          ieee_divide(GREATEST(LENGTH(`t0`.`string_col`), 10) - LENGTH(`t0`.`string_col`), 2)
        ) AS INT64)
      ) + LENGTH(`t0`.`string_col`),
      '-'
    ),
    GREATEST(LENGTH(`t0`.`string_col`), 10),
    '-'
  ) AS `both`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`