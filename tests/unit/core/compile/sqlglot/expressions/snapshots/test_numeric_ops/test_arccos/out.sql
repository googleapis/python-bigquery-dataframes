SELECT
  IF(
    NOT (
      ABS(`t0`.`float64_col`) <= 1
    ),
    CAST('NaN' AS FLOAT64),
    ACOS(`t0`.`float64_col`)
  ) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`