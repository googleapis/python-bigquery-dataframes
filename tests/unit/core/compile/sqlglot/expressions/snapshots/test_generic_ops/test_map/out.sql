WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  CASE
    WHEN `string_col` = 'value1'
    THEN 'mapped1'
    WHEN `string_col` IS NULL
    THEN 'UNKNOWN'
    ELSE `string_col`
  END AS `string_col`
FROM `bfcte_0`