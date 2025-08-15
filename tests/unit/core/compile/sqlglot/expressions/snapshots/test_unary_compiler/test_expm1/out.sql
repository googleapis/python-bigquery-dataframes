SELECT
  IF(
    NOT (
      `t0`.`float64_col` < 709.78
    ),
    CAST('Infinity' AS FLOAT64),
    EXP(`t0`.`float64_col`)
  ) - 1 AS `float64_col`
FROM (
  SELECT
    `float64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
) AS `t0`