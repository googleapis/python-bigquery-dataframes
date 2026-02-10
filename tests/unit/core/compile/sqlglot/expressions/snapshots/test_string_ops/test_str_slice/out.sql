SELECT
  SUBSTRING(
    `t0`.`string_col`,
    IF(
      (
        IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1
      ) >= 1,
      IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1,
      IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1)) + 1 + LENGTH(`t0`.`string_col`)
    ),
    GREATEST(
      0,
      IF(3 >= 0, 3, GREATEST(0, LENGTH(`t0`.`string_col`) + 3)) - IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1))
    )
  ) AS `string_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`