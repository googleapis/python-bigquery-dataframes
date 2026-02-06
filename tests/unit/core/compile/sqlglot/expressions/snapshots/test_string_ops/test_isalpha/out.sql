WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  REGEXP_CONTAINS(`string_col`, '^\\p{L}+$') AS `string_col`
FROM `bfcte_0`