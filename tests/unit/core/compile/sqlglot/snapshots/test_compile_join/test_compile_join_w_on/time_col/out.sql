WITH `bfcte_0` AS (
  SELECT
    `rowindex` AS `bfcol_2`,
    `time_col` AS `bfcol_3`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
), `bfcte_1` AS (
  SELECT
    `rowindex` AS `bfcol_0`,
    `time_col` AS `bfcol_1`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
), `bfcte_2` AS (
  SELECT
    `bfcte_0`.*
  FROM `bfcte_0`
), `bfcte_3` AS (
  SELECT
    `bfcte_1`.*
  FROM `bfcte_1`
)
SELECT
  `bfcol_0` AS `rowindex_x`,
  `bfcol_1` AS `time_col`,
  `bfcol_4` AS `rowindex_y`
FROM (
  SELECT
    *
  FROM `bfcte_3`
  INNER JOIN (
    SELECT
      `bfcol_2` AS `bfcol_4`,
      `bfcol_3` AS `bfcol_5`
    FROM `bfcte_2`
  )
    ON COALESCE(CAST(`bfcol_1` AS STRING), '0') = COALESCE(CAST(`bfcol_5` AS STRING), '0')
    AND COALESCE(CAST(`bfcol_1` AS STRING), '1') = COALESCE(CAST(`bfcol_5` AS STRING), '1')
)