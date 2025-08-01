SELECT
  IF(
    NOT (
      `t0`.`float64_col` >= 1
    ),
    CAST('NaN' AS FLOAT64),
    LN(`t0`.`float64_col` + SQRT((
      `t0`.`float64_col` * `t0`.`float64_col`
    ) - 1))
  ) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`