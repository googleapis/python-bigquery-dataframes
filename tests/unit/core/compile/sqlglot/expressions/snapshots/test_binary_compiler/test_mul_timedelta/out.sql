SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`int64_col`,
  CAST(FLOOR(86400000000 * `t0`.`int64_col`) AS INT64) AS `timedelta_mul_numeric`,
  CAST(FLOOR(`t0`.`int64_col` * 86400000000) AS INT64) AS `numeric_mul_timedelta`
FROM (
  SELECT
    `int64_col`,
    `rowindex`,
    `timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
) AS `t0`