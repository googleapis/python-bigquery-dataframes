SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`date_col`,
  TIMESTAMP_ADD(CAST(`t0`.`date_col` AS DATETIME), INTERVAL 86400000000 MICROSECOND) AS `date_add_timedelta`,
  TIMESTAMP_ADD(`t0`.`timestamp_col`, INTERVAL 86400000000 MICROSECOND) AS `timestamp_add_timedelta`,
  TIMESTAMP_ADD(CAST(`t0`.`date_col` AS DATETIME), INTERVAL 86400000000 MICROSECOND) AS `timedelta_add_date`,
  TIMESTAMP_ADD(`t0`.`timestamp_col`, INTERVAL 86400000000 MICROSECOND) AS `timedelta_add_timestamp`,
  172800000000 AS `timedelta_add_timedelta`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`