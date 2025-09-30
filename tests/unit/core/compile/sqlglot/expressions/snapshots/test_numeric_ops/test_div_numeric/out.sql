SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`bool_col`,
  `t0`.`float64_col`,
  ieee_divide(`t0`.`int64_col`, `t0`.`int64_col`) AS `int_div_int`,
  ieee_divide(`t0`.`int64_col`, 1) AS `int_div_1`,
  ieee_divide(`t0`.`int64_col`, 0.0) AS `int_div_0`,
  ieee_divide(`t0`.`int64_col`, `t0`.`float64_col`) AS `int_div_float`,
  ieee_divide(`t0`.`float64_col`, `t0`.`int64_col`) AS `float_div_int`,
  ieee_divide(`t0`.`float64_col`, 0.0) AS `float_div_0`,
  ieee_divide(`t0`.`int64_col`, CAST(`t0`.`bool_col` AS INT64)) AS `int_div_bool`,
  ieee_divide(CAST(`t0`.`bool_col` AS INT64), `t0`.`int64_col`) AS `bool_div_int`
FROM (
  SELECT
    `bool_col`,
    `int64_col`,
    `float64_col`,
    `rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`