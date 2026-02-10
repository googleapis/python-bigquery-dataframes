SELECT
  CASE
    WHEN `t1`.`int64_col` < (
      MIN(`t1`.`int64_col`) OVER () + (
        (
          ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
        ) * 1
      )
    )
    THEN 'a'
    WHEN `t1`.`int64_col` < (
      MIN(`t1`.`int64_col`) OVER () + (
        (
          ieee_divide(MAX(`t1`.`int64_col`) OVER () - MIN(`t1`.`int64_col`) OVER (), 3)
        ) * 2
      )
    )
    THEN 'b'
    WHEN (
      `t1`.`int64_col`
    ) IS NOT NULL
    THEN 'c'
    ELSE CAST(NULL AS STRING)
  END AS `int_bins_labels`
FROM (
  SELECT
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`