SELECT
  LN(
    ABS(`t0`.`float64_col`) + SQRT((
      `t0`.`float64_col` * `t0`.`float64_col`
    ) + 1)
  ) * SIGN(`t0`.`float64_col`) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`