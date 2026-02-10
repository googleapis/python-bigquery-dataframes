SELECT
  CASE
    WHEN (
      `t0`.`float64_col` IS NULL
    ) OR (
      `t0`.`int64_col` < `t0`.`float64_col`
    )
    THEN `t0`.`float64_col`
    ELSE `t0`.`int64_col`
  END AS `int64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`