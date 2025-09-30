SELECT
  IF(
    NOT (
      ABS(`t0`.`float64_col`) < 1
    ),
    IF(
      ABS(`t0`.`float64_col`) = 1,
      CAST('Infinity' AS FLOAT64) * `t0`.`float64_col`,
      CAST('NaN' AS FLOAT64)
    ),
    ieee_divide(LN(ieee_divide(`t0`.`float64_col` + 1, 1 - `t0`.`float64_col`)), 2)
  ) AS `float64_col`
FROM (
  SELECT
    `float64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`