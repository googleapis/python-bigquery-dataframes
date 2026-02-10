SELECT
  CONCAT(
    UPPER(
      SUBSTRING(`t0`.`string_col`, IF((
        0 + 1
      ) >= 1, 0 + 1, 0 + 1 + LENGTH(`t0`.`string_col`)), 1)
    ),
    LOWER(
      SUBSTRING(
        `t0`.`string_col`,
        IF((
          1 + 1
        ) >= 1, 1 + 1, 1 + 1 + LENGTH(`t0`.`string_col`)),
        LENGTH(`t0`.`string_col`)
      )
    )
  ) AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`