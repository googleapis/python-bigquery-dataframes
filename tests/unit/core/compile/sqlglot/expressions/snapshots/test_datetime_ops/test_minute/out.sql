SELECT
  EXTRACT(minute FROM `t0`.`timestamp_col`) AS `timestamp_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`