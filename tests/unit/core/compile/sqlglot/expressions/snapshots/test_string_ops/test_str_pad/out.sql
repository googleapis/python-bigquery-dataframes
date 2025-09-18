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
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
) AS `t0`