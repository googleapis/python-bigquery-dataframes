WITH `bfcte_0` AS (
  SELECT
    `rowindex` AS `bfcol_0`,
    `rowindex_2` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    ST_GEOGPOINT(`bfcol_0`, `bfcol_1`) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `rowindex`
FROM `bfcte_1`