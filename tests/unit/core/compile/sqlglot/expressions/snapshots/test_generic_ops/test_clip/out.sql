SELECT
  GREATEST(LEAST(`t0`.`rowindex`, `t0`.`int64_too`), `t0`.`int64_col`) AS `result_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`