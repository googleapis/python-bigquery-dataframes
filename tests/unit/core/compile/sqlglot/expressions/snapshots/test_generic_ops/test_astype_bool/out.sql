WITH `bfcte_0` AS (
  SELECT
    `bool_col`,
    `float64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  `bool_col` AS `bool_col`,
  `float64_col` <> 0 AS `float64_col`,
  `float64_col` <> 0 AS `float64_w_safe`
FROM `bfcte_0`