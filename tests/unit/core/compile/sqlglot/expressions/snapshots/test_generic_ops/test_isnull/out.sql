WITH `bfcte_0` AS (
  SELECT
    `float64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  `float64_col` IS NULL AS `float64_col`
FROM `bfcte_0`