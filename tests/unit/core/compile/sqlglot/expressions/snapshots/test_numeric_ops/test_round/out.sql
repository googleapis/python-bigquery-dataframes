SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`float64_col`,
  CAST(trunc(ROUND(`t0`.`int64_col`, 0)) AS INT64) AS `int_round_0`,
  CAST(trunc(ROUND(`t0`.`int64_col`, 1)) AS INT64) AS `int_round_1`,
  CAST(trunc(ROUND(`t0`.`int64_col`, -1)) AS INT64) AS `int_round_m1`,
  ROUND(`t0`.`float64_col`, 0) AS `float_round_0`,
  ROUND(`t0`.`float64_col`, 1) AS `float_round_1`,
  ROUND(`t0`.`float64_col`, -1) AS `float_round_m1`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`