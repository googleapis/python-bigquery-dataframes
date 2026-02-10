SELECT
  CAST(TIMESTAMP_SUB(
    DATETIME(
      IF(
        (
          (
            (
              MOD(
                (
                  `t0`.`rowindex` * 1
                ) + (
                  (
                    (
                      EXTRACT(year FROM `t0`.`timestamp_col`) * 4
                    ) + EXTRACT(quarter FROM `t0`.`timestamp_col`)
                  ) - 1
                ),
                4
              )
            ) + 1
          ) * 3
        ) = 12,
        (
          CAST(FLOOR(
            ieee_divide(
              (
                `t0`.`rowindex` * 1
              ) + (
                (
                  (
                    EXTRACT(year FROM `t0`.`timestamp_col`) * 4
                  ) + EXTRACT(quarter FROM `t0`.`timestamp_col`)
                ) - 1
              ),
              4
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
                  EXTRACT(year FROM `t0`.`timestamp_col`) * 4
                ) + EXTRACT(quarter FROM `t0`.`timestamp_col`)
              ) - 1
            ),
            4
          )
        ) AS INT64)
      ),
      IF(
        (
          (
            (
              MOD(
                (
                  `t0`.`rowindex` * 1
                ) + (
                  (
                    (
                      EXTRACT(year FROM `t0`.`timestamp_col`) * 4
                    ) + EXTRACT(quarter FROM `t0`.`timestamp_col`)
                  ) - 1
                ),
                4
              )
            ) + 1
          ) * 3
        ) = 12,
        1,
        (
          (
            (
              MOD(
                (
                  `t0`.`rowindex` * 1
                ) + (
                  (
                    (
                      EXTRACT(year FROM `t0`.`timestamp_col`) * 4
                    ) + EXTRACT(quarter FROM `t0`.`timestamp_col`)
                  ) - 1
                ),
                4
              )
            ) + 1
          ) * 3
        ) + 1
      ),
      1,
      0,
      0,
      0
    ),
    INTERVAL '1' DAY
  ) AS TIMESTAMP) AS `non_fixed_freq`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`