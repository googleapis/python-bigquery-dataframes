SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`int64_col`,
  CASE
    WHEN (
      ieee_divide(86400000000, `t0`.`int64_col`)
    ) > 0
    THEN CAST(FLOOR(ieee_divide(86400000000, `t0`.`int64_col`)) AS INT64)
    ELSE CAST(CEIL(ieee_divide(86400000000, `t0`.`int64_col`)) AS INT64)
  END AS `timedelta_div_numeric`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`