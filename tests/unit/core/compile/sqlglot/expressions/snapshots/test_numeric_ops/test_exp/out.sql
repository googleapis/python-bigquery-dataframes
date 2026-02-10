SELECT
  IF(
    NOT (
      `t0`.`float64_col` < 709.78
    ),
    CAST('Infinity' AS FLOAT64),
    EXP(`t0`.`float64_col`)
  ) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`