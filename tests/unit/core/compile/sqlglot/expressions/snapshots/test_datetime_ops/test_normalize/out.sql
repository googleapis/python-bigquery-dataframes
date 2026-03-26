SELECT
  CAST(TIMESTAMP_TRUNC(`t0`.`timestamp_col`, DAY) AS TIMESTAMP) AS `timestamp_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`