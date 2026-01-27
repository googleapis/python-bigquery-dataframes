SELECT
  `t0`.`int64_col`,
  COALESCE(`t0`.`int64_too`, `t0`.`int64_col`) AS `int64_too`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`