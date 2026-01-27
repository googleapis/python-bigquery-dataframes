SELECT
  CASE
    WHEN `t1`.`int64_col` <= (
      MIN(`t1`.`int64_col`) OVER () + (
        (
          ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
        ) * 1
      )
    )
    THEN STRUCT(
      (
        MIN(`t1`.`int64_col`) OVER () + (
          (
            ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
          ) * 0
        )
      ) - (
        (
          MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER ()
        ) * 0.001
      ) AS `left_exclusive`,
      (
        MIN(`t1`.`int64_col`) OVER () + (
          (
            ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
          ) * 1
        )
      ) + 0 AS `right_inclusive`
    )
    WHEN `t1`.`int64_col` <= (
      MIN(`t1`.`int64_col`) OVER () + (
        (
          ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
        ) * 2
      )
    )
    THEN STRUCT(
      (
        MIN(`t1`.`int64_col`) OVER () + (
          (
            ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
          ) * 1
        )
      ) - 0 AS `left_exclusive`,
      (
        MIN(`t1`.`int64_col`) OVER () + (
          (
            ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
          ) * 2
        )
      ) + 0 AS `right_inclusive`
    )
    WHEN (
      `t1`.`int64_col`
    ) IS NOT NULL
    THEN STRUCT(
      (
        MIN(`t1`.`int64_col`) OVER () + (
          (
            ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
          ) * 2
        )
      ) - 0 AS `left_exclusive`,
      (
        MIN(`t1`.`int64_col`) OVER () + (
          (
            ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
          ) * 3
        )
      ) + 0 AS `right_inclusive`
    )
    ELSE CAST(NULL AS STRUCT<`left_exclusive` FLOAT64, `right_inclusive` FLOAT64>)
  END AS `int_bins`
FROM (
  SELECT
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`