WITH `bfcte_0` AS (
  SELECT
    `rowindex`,
    `rowindex_2`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  ST_GEOGPOINT(`rowindex`, `rowindex_2`) AS `rowindex`
FROM `bfcte_0`