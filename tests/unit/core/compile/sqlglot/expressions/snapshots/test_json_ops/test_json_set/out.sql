WITH `bfcte_0` AS (
  SELECT
    `json_col`
  FROM `bigframes-dev`.`sqlglot_test`.`json_types`
)
SELECT
  *,
  JSON_SET(`json_col`, '$.a', 100) AS `json_col`
FROM `bfcte_0`