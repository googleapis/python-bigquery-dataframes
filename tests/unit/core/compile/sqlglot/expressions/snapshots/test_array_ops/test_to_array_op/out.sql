SELECT
  [COALESCE(`t0`.`bool_col`, FALSE)] AS `bool_col`,
  [COALESCE(`t0`.`int64_col`, 0)] AS `int64_col`,
  [COALESCE(`t0`.`string_col`, ''), COALESCE(`t0`.`string_col`, '')] AS `strs_col`,
  [
    COALESCE(`t0`.`int64_col`, 0),
    CAST(COALESCE(`t0`.`bool_col`, FALSE) AS INT64),
    COALESCE(`t0`.`float64_col`, 0.0)
  ] AS `numeric_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`