WITH `bfcte_0` AS (
  SELECT
    `bool_col` AS `bfcol_4`,
    `rowindex` AS `bfcol_5`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
), `bfcte_1` AS (
  SELECT
    `bool_col` AS `bfcol_0`,
    `rowindex` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
), `bfcte_2` AS (
  SELECT
    `bfcte_0`.*
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `rowindex_x`,
  `bfcol_3` AS `bool_col`,
  `bfcol_6` AS `rowindex_y`
FROM (
  SELECT
    *
  FROM (
    SELECT
      `bfcol_1` AS `bfcol_2`,
      `bfcol_0` AS `bfcol_3`
    FROM `bfcte_1`
  )
  INNER JOIN (
    SELECT
      `bfcol_5` AS `bfcol_6`,
      `bfcol_4` AS `bfcol_7`
    FROM `bfcte_2`
  )
    ON COALESCE(CAST(`bfcol_3` AS STRING), '0') = COALESCE(CAST(`bfcol_7` AS STRING), '0')
    AND COALESCE(CAST(`bfcol_3` AS STRING), '1') = COALESCE(CAST(`bfcol_7` AS STRING), '1')
)