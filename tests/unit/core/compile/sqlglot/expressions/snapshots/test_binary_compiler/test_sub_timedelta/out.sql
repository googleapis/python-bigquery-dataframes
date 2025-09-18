SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64) AS `duration_col`,
  `t0`.`date_col`,
  TIMESTAMP_SUB(
    CAST(`t0`.`date_col` AS DATETIME),
    INTERVAL (CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64)) MICROSECOND
  ) AS `date_sub_timedelta`,
  TIMESTAMP_SUB(
    `t0`.`timestamp_col`,
    INTERVAL (CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64)) MICROSECOND
  ) AS `timestamp_sub_timedelta`,
  DATE_DIFF(`t0`.`date_col`, `t0`.`date_col`, DAY) * 86400000000 AS `timestamp_sub_date`,
  TIMESTAMP_DIFF(`t0`.`timestamp_col`, `t0`.`timestamp_col`, MICROSECOND) AS `date_sub_timestamp`,
  CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64) - CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64) AS `timedelta_sub_timedelta`
FROM (
  SELECT
    `date_col`,
    `rowindex`,
    `timestamp_col`,
    `duration_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
) AS `t0`