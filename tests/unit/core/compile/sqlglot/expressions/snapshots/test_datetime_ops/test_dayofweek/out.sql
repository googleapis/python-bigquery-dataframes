SELECT
  CAST(MOD(EXTRACT(dayofweek FROM `t0`.`datetime_col`) + 5, 7) AS INT64) AS `datetime_col`,
  CAST(MOD(EXTRACT(dayofweek FROM `t0`.`timestamp_col`) + 5, 7) AS INT64) AS `timestamp_col`,
  CAST(MOD(EXTRACT(dayofweek FROM `t0`.`date_col`) + 5, 7) AS INT64) AS `date_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`