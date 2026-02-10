SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`int64_col`,
  CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64) AS `duration_col`,
  CAST(FLOOR(CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64) * `t0`.`int64_col`) AS INT64) AS `timedelta_mul_numeric`,
  CAST(FLOOR(`t0`.`int64_col` * CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64)) AS INT64) AS `numeric_mul_timedelta`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`