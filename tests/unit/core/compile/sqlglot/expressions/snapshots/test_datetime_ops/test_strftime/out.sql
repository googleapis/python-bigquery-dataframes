SELECT
  format_date('%Y-%m-%d', `t0`.`date_col`) AS `date_col`,
  format_datetime('%Y-%m-%d', `t0`.`datetime_col`) AS `datetime_col`,
  format_time('%Y-%m-%d', `t0`.`time_col`) AS `time_col`,
  format_timestamp('%Y-%m-%d', `t0`.`timestamp_col`, 'UTC') AS `timestamp_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`