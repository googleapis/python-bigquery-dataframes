WITH `bfcte_0` AS (
  SELECT
    COUNT(1) AS `bfcol_32`
  FROM (
    SELECT
      *
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)
SELECT
  `bfcol_32` AS `size`
FROM `bfcte_0`