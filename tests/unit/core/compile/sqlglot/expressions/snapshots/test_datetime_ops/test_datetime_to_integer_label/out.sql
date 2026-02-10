SELECT
  CAST(FLOOR(
    ieee_divide(
      UNIX_MICROS(CAST(`t0`.`datetime_col` AS TIMESTAMP)) - UNIX_MICROS(`t0`.`timestamp_col`),
      86400000000.0
    )
  ) AS INT64) AS `fixed_freq`,
  CASE
    WHEN UNIX_MICROS(
      CAST(TIMESTAMP_ADD(TIMESTAMP_TRUNC(`t0`.`datetime_col`, WEEK(MONDAY)), INTERVAL '6' DAY) AS TIMESTAMP)
    ) = UNIX_MICROS(
      CAST(TIMESTAMP_ADD(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, WEEK(MONDAY)), INTERVAL '6' DAY) AS TIMESTAMP)
    )
    THEN 0
    ELSE (
      CAST(FLOOR(
        ieee_divide(
          (
            UNIX_MICROS(
              CAST(TIMESTAMP_ADD(TIMESTAMP_TRUNC(`t0`.`datetime_col`, WEEK(MONDAY)), INTERVAL '6' DAY) AS TIMESTAMP)
            ) - UNIX_MICROS(
              CAST(TIMESTAMP_ADD(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, WEEK(MONDAY)), INTERVAL '6' DAY) AS TIMESTAMP)
            )
          ) - 1,
          604800000000
        )
      ) AS INT64)
    ) + 1
  END AS `non_fixed_freq_weekly`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`