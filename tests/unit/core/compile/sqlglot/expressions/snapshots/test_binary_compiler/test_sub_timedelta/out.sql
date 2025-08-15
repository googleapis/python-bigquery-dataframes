SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`date_col`,
  TIMESTAMP_SUB(CAST(`t0`.`date_col` AS DATETIME), INTERVAL 86400000000 MICROSECOND) AS `date_sub_timedelta`,
  TIMESTAMP_SUB(`t0`.`timestamp_col`, INTERVAL 86400000000 MICROSECOND) AS `timestamp_sub_timedelta`,
  DATE_DIFF(`t0`.`date_col`, `t0`.`date_col`, DAY) * 86400000000 AS `timestamp_sub_date`,
  TIMESTAMP_DIFF(`t0`.`timestamp_col`, `t0`.`timestamp_col`, MICROSECOND) AS `date_sub_timestamp`,
  0 AS `timedelta_sub_timedelta`
FROM (
  SELECT
    `date_col`,
    `rowindex`,
    `timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
) AS `t0`