SELECT
  NOT (
    IS_INF(`t0`.`float64_col`)
  ) AND NOT (
    IS_NAN(`t0`.`float64_col`)
  ) AS `float64_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`