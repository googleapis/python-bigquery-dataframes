SELECT
  IF(
    NOT (
      ABS(`t0`.`float64_col`) < 20
    ),
    SIGN(`t0`.`float64_col`),
    ieee_divide(
      EXP(`t0`.`float64_col`) - EXP(-(
        `t0`.`float64_col`
      )),
      EXP(`t0`.`float64_col`) + EXP(-(
        `t0`.`float64_col`
      ))
    )
  ) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`