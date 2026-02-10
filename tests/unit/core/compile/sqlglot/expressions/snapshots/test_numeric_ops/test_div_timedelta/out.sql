SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`int64_col`,
  CAST(FLOOR(ieee_divide(86400000000, `t0`.`int64_col`)) AS INT64) AS `timedelta_div_numeric`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`