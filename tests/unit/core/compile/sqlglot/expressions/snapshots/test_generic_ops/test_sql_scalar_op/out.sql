SELECT
  CAST(`t0`.`bool_col` AS INT64) + BYTE_LENGTH(`t0`.`bytes_col`) AS `bool_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`