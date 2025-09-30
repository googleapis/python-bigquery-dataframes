SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`int64_col`,
  CAST(FLOOR(ieee_divide(86400000000, `t0`.`int64_col`)) AS INT64) AS `timedelta_div_numeric`
FROM (
  SELECT
    `int64_col`,
    `rowindex`,
    `timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`