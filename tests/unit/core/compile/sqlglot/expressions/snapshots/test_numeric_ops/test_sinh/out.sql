SELECT
  IF(
    NOT (
      ABS(`t0`.`float64_col`) < 709.78
    ),
    CAST('Infinity' AS FLOAT64) * SIGN(`t0`.`float64_col`),
    ieee_divide(EXP(`t0`.`float64_col`) - EXP(-(
      `t0`.`float64_col`
    )), 2)
  ) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`