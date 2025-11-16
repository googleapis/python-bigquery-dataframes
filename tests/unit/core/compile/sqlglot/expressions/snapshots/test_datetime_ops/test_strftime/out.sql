SELECT
  format_timestamp('%Y-%m-%d', `t0`.`timestamp_col`, 'UTC') AS `timestamp_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`