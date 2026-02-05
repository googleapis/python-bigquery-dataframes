WITH `bfcte_0` AS (
  SELECT
    `bool_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  `bool_col` <> LAG(`bool_col`, 1) OVER (ORDER BY `bool_col` DESC) AS `diff_bool`
FROM `bfcte_0`