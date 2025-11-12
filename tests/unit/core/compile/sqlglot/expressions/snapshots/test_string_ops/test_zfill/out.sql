SELECT
  CASE
    WHEN SUBSTRING(`t0`.`string_col`, IF((
      0 + 1
    ) >= 1, 0 + 1, 0 + 1 + LENGTH(`t0`.`string_col`)), 1) = '-'
    THEN CONCAT(
      '-',
      LPAD(
        SUBSTRING(`t0`.`string_col`, IF((
          1 + 1
        ) >= 1, 1 + 1, 1 + 1 + LENGTH(`t0`.`string_col`))),
        GREATEST(
          LENGTH(
            SUBSTRING(`t0`.`string_col`, IF((
              1 + 1
            ) >= 1, 1 + 1, 1 + 1 + LENGTH(`t0`.`string_col`)))
          ),
          9
        ),
        '0'
      )
    )
    ELSE LPAD(`t0`.`string_col`, GREATEST(LENGTH(`t0`.`string_col`), 10), '0')
  END AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`