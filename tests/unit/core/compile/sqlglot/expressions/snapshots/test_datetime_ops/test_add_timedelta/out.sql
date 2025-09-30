SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`date_col`,
  TIMESTAMP_ADD(CAST(`t0`.`date_col` AS DATETIME), INTERVAL 86400000000 MICROSECOND) AS `date_add_timedelta`,
  TIMESTAMP_ADD(`t0`.`timestamp_col`, INTERVAL 86400000000 MICROSECOND) AS `timestamp_add_timedelta`,
  TIMESTAMP_ADD(CAST(`t0`.`date_col` AS DATETIME), INTERVAL 86400000000 MICROSECOND) AS `timedelta_add_date`,
  TIMESTAMP_ADD(`t0`.`timestamp_col`, INTERVAL 86400000000 MICROSECOND) AS `timedelta_add_timestamp`,
  172800000000 AS `timedelta_add_timedelta`
FROM (
  SELECT
    `date_col`,
    `rowindex`,
    `timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`