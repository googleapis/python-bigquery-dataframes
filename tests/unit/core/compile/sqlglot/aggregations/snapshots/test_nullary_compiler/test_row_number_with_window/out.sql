WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  ROW_NUMBER() OVER (ORDER BY `int64_col` ASC NULLS LAST) - 1 AS `row_number`
FROM `bfcte_0`