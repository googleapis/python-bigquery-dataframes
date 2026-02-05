WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  REPEAT(`string_col`, 2) AS `string_col`
FROM `bfcte_0`