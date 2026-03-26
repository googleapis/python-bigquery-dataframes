SELECT
  CAST(TIMESTAMP_SUB(
    DATETIME(
      IF(
        (
          (
            MOD(
              (
                `t0`.`rowindex` * 1
              ) + (
                (
                  (
                    EXTRACT(year FROM `t0`.`timestamp_col`) * 12
                  ) + EXTRACT(month FROM `t0`.`timestamp_col`)
                ) - 1
              ),
              12
            )
          ) + 1
        ) = 12,
        (
          CAST(FLOOR(
            ieee_divide(
              (
                `t0`.`rowindex` * 1
              ) + (
                (
                  (
                    EXTRACT(year FROM `t0`.`timestamp_col`) * 12
                  ) + EXTRACT(month FROM `t0`.`timestamp_col`)
                ) - 1
              ),
              12
            )
          ) AS INT64)
        ) + 1,
        CAST(FLOOR(
          ieee_divide(
            (
              `t0`.`rowindex` * 1
            ) + (
              (
                (
                  EXTRACT(year FROM `t0`.`timestamp_col`) * 12
                ) + EXTRACT(month FROM `t0`.`timestamp_col`)
              ) - 1
            ),
            12
          )
        ) AS INT64)
      ),
      IF(
        (
          (
            MOD(
              (
                `t0`.`rowindex` * 1
              ) + (
                (
                  (
                    EXTRACT(year FROM `t0`.`timestamp_col`) * 12
                  ) + EXTRACT(month FROM `t0`.`timestamp_col`)
                ) - 1
              ),
              12
            )
          ) + 1
        ) = 12,
        1,
        (
          (
            MOD(
              (
                `t0`.`rowindex` * 1
              ) + (
                (
                  (
                    EXTRACT(year FROM `t0`.`timestamp_col`) * 12
                  ) + EXTRACT(month FROM `t0`.`timestamp_col`)
                ) - 1
              ),
              12
            )
          ) + 1
        ) + 1
      ),
      1,
      0,
      0,
      0
    ),
    INTERVAL '1' DAY
  ) AS TIMESTAMP) AS `non_fixed_freq_monthly`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`