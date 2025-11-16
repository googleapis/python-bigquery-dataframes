SELECT
  IF(
    NOT (
      (
        1 + `t0`.`float64_col`
      ) > 0
    ),
    IF(
      (
        1 + `t0`.`float64_col`
      ) = 0,
      CAST('-Infinity' AS FLOAT64),
      CAST('NaN' AS FLOAT64)
    ),
    LN(1 + `t0`.`float64_col`)
  ) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`