SELECT
  UNIX_MILLIS(`t1`.`timestamp_col`) AS `timestamp_col`
FROM (
  SELECT
    `t0`.`timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`