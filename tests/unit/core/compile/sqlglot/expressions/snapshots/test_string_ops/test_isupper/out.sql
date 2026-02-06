WITH `bfcte_0` AS (
  SELECT
    `string_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  UPPER(`string_col`) = `string_col` AND LOWER(`string_col`) <> `string_col` AS `string_col`
FROM `bfcte_0`