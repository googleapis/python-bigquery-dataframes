WITH `bfcte_0` AS (
  SELECT
    `json_col` AS `bfcol_0`
  FROM `bigframes-dev`.`sqlglot_test`.`json_types`
), `bfcte_1` AS (
  SELECT
    *,
    JSON_SET(`bfcol_0`, '$.a', 100) AS `bfcol_1`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `json_col`
FROM `bfcte_1`