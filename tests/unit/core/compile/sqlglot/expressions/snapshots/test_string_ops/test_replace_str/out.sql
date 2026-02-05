WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  REPLACE(`string_col`, 'e', 'a') AS `string_col`
FROM `bfcte_0`