SELECT
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, MICROSECOND) AS TIMESTAMP) AS `timestamp_col_us`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, MILLISECOND) AS TIMESTAMP) AS `timestamp_col_ms`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, SECOND) AS TIMESTAMP) AS `timestamp_col_s`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, MINUTE) AS TIMESTAMP) AS `timestamp_col_min`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, HOUR) AS TIMESTAMP) AS `timestamp_col_h`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, DAY) AS TIMESTAMP) AS `timestamp_col_D`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, WEEK(MONDAY)) AS TIMESTAMP) AS `timestamp_col_W`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, MONTH) AS TIMESTAMP) AS `timestamp_col_M`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, QUARTER) AS TIMESTAMP) AS `timestamp_col_Q`,
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, YEAR) AS TIMESTAMP) AS `timestamp_col_Y`,
  TIMESTAMP_TRUNC(`t0`.`datetime_col`, MICROSECOND) AS `datetime_col_q`,
  TIMESTAMP_TRUNC(`t0`.`datetime_col`, MICROSECOND) AS `datetime_col_us`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`