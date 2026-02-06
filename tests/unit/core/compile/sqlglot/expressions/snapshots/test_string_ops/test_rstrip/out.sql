WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  RTRIM(`string_col`, ' ') AS `string_col`
FROM `bfcte_0`