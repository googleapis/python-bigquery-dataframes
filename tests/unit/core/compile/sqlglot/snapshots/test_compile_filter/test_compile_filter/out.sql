WITH `bfcte_0` AS (
  SELECT
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  `rowindex`,
  `rowindex` AS `rowindex_1`,
  `int64_col`
FROM `bfcte_0`