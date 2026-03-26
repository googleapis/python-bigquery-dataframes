SELECT
  CAST(FLOOR(
    ieee_divide(
      UNIX_MICROS(CAST(`t0`.`datetime_col` AS TIMESTAMP)) - UNIX_MICROS(`t0`.`timestamp_col`),
      86400000000.0
    )
  ) AS INT64) AS `fixed_freq`,
  CAST(FLOOR(
    ieee_divide(UNIX_MICROS(CAST(`t0`.`datetime_col` AS TIMESTAMP)) - 0, 86400000000.0)
  ) AS INT64) AS `origin_epoch`,
  CAST(FLOOR(
    ieee_divide(
      UNIX_MICROS(CAST(`t0`.`datetime_col` AS TIMESTAMP)) - UNIX_MICROS(CAST(CAST(`t0`.`timestamp_col` AS DATE) AS TIMESTAMP)),
      86400000000.0
    )
  ) AS INT64) AS `origin_start_day`,
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
  END AS `non_fixed_freq_weekly`,
  CASE
    WHEN (
      (
        (
          EXTRACT(year FROM `t0`.`datetime_col`) * 12
        ) + EXTRACT(month FROM `t0`.`datetime_col`)
      ) - 1
    ) = (
      (
        (
          EXTRACT(year FROM `t0`.`timestamp_col`) * 12
        ) + EXTRACT(month FROM `t0`.`timestamp_col`)
      ) - 1
    )
    THEN 0
    ELSE (
      CAST(FLOOR(
        ieee_divide(
          (
            (
              (
                (
                  EXTRACT(year FROM `t0`.`datetime_col`) * 12
                ) + EXTRACT(month FROM `t0`.`datetime_col`)
              ) - 1
            ) - (
              (
                (
                  EXTRACT(year FROM `t0`.`timestamp_col`) * 12
                ) + EXTRACT(month FROM `t0`.`timestamp_col`)
              ) - 1
            )
          ) - 1,
          1
        )
      ) AS INT64)
    ) + 1
  END AS `non_fixed_freq_monthly`,
  CASE
    WHEN (
      (
        (
          EXTRACT(year FROM `t0`.`datetime_col`) * 4
        ) + EXTRACT(quarter FROM `t0`.`datetime_col`)
      ) - 1
    ) = (
      (
        (
          EXTRACT(year FROM `t0`.`timestamp_col`) * 4
        ) + EXTRACT(quarter FROM `t0`.`timestamp_col`)
      ) - 1
    )
    THEN 0
    ELSE (
      CAST(FLOOR(
        ieee_divide(
          (
            (
              (
                (
                  EXTRACT(year FROM `t0`.`datetime_col`) * 4
                ) + EXTRACT(quarter FROM `t0`.`datetime_col`)
              ) - 1
            ) - (
              (
                (
                  EXTRACT(year FROM `t0`.`timestamp_col`) * 4
                ) + EXTRACT(quarter FROM `t0`.`timestamp_col`)
              ) - 1
            )
          ) - 1,
          1
        )
      ) AS INT64)
    ) + 1
  END AS `non_fixed_freq_quarterly`,
  CASE
    WHEN EXTRACT(year FROM `t0`.`datetime_col`) = EXTRACT(year FROM `t0`.`timestamp_col`)
    THEN 0
    ELSE (
      CAST(FLOOR(
        ieee_divide(
          (
            EXTRACT(year FROM `t0`.`datetime_col`) - EXTRACT(year FROM `t0`.`timestamp_col`)
          ) - 1,
          1
        )
      ) AS INT64)
    ) + 1
  END AS `non_fixed_freq_yearly`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`