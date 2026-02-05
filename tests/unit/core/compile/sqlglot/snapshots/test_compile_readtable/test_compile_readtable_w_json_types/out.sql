WITH `bfcte_0` AS (
  SELECT
    `json_col`,
    `rowindex`
  FROM `bigframes-dev`.`sqlglot_test`.`json_types`
)
SELECT
  *,
  `rowindex` AS `rowindex`,
  `json_col` AS `json_col`
FROM `bfcte_0`