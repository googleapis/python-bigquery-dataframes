SELECT
  COALESCE(`t0`.`int64_col`, `t0`.`float64_col`) AS `int64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`