WITH `bfcte_0` AS (
  SELECT
    `people`
  FROM `bigframes-dev`.`sqlglot_test`.`nested_structs_types`
)
SELECT
  `people`.`name` AS `string`,
  `people`.`name` AS `int`
FROM `bfcte_0`