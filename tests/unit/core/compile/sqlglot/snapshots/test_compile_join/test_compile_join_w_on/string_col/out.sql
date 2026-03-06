WITH `bfcte_0` AS (
  SELECT
  FROM `bfcte_0`
), `bfcte_1` AS (
  SELECT
  FROM `bfcte_1`
), `bfcte_2` AS (
  SELECT
  FROM `bfcte_2`
), `bfcte_3` AS (
  SELECT
  FROM `bfcte_3`
)
SELECT
FROM (
  SELECT
    `bfcol_0` AS `rowindex_x`,
    `bfcol_1` AS `string_col`,
    `bfcol_4` AS `rowindex_y`
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