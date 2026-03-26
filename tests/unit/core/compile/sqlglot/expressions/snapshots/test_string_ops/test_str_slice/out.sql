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
  ) AS `1_3`,
  SUBSTRING(
    `t0`.`string_col`,
    IF((
      0 + 1
    ) >= 1, 0 + 1, 0 + 1 + LENGTH(`t0`.`string_col`)),
    GREATEST(0, IF(3 >= 0, 3, GREATEST(0, LENGTH(`t0`.`string_col`) + 3)) - 0)
  ) AS `none_3`,
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
      LENGTH(`t0`.`string_col`) - IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1))
    )
  ) AS `1_none`,
  SUBSTRING(
    `t0`.`string_col`,
    IF(
      (
        IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) + 1
      ) >= 1,
      IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) + 1,
      IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) + 1 + LENGTH(`t0`.`string_col`)
    ),
    GREATEST(
      0,
      LENGTH(`t0`.`string_col`) - IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3))
    )
  ) AS `m3_none`,
  SUBSTRING(
    `t0`.`string_col`,
    IF((
      0 + 1
    ) >= 1, 0 + 1, 0 + 1 + LENGTH(`t0`.`string_col`)),
    GREATEST(0, IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) - 0)
  ) AS `none_m3`,
  SUBSTRING(
    `t0`.`string_col`,
    IF(
      (
        IF(-5 >= 0, -5, GREATEST(0, LENGTH(`t0`.`string_col`) + -5)) + 1
      ) >= 1,
      IF(-5 >= 0, -5, GREATEST(0, LENGTH(`t0`.`string_col`) + -5)) + 1,
      IF(-5 >= 0, -5, GREATEST(0, LENGTH(`t0`.`string_col`) + -5)) + 1 + LENGTH(`t0`.`string_col`)
    ),
    GREATEST(
      0,
      IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) - IF(-5 >= 0, -5, GREATEST(0, LENGTH(`t0`.`string_col`) + -5))
    )
  ) AS `m5_m3`,
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
      IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) - IF(1 >= 0, 1, GREATEST(0, LENGTH(`t0`.`string_col`) + 1))
    )
  ) AS `1_m3`,
  SUBSTRING(
    `t0`.`string_col`,
    IF(
      (
        IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) + 1
      ) >= 1,
      IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) + 1,
      IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3)) + 1 + LENGTH(`t0`.`string_col`)
    ),
    GREATEST(
      0,
      IF(5 >= 0, 5, GREATEST(0, LENGTH(`t0`.`string_col`) + 5)) - IF(-3 >= 0, -3, GREATEST(0, LENGTH(`t0`.`string_col`) + -3))
    )
  ) AS `m3_5`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`