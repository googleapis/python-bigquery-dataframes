WITH `bfcte_0` AS (
  SELECT
    `people` AS `bfcol_0`
  FROM `bigframes-dev`.`sqlglot_test`.`nested_structs_types`
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_0`.`name` AS `bfcol_1`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `people`
FROM `bfcte_1`