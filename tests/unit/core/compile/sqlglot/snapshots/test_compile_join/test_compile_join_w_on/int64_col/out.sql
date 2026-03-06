WITH `bfcte_0` AS (
  SELECT
  FROM `bfcte_0`
), `bfcte_1` AS (
  SELECT
  FROM `bfcte_1`
), `bfcte_2` AS (
  SELECT
  FROM `bfcte_2`
)
SELECT
FROM (
  SELECT
    `bfcol_2` AS `rowindex_x`,
    `bfcol_3` AS `int64_col`,
    `bfcol_6` AS `rowindex_y`
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
    ON COALESCE(`bfcol_3`, 0) = COALESCE(`bfcol_7`, 0)
    AND COALESCE(`bfcol_3`, 1) = COALESCE(`bfcol_7`, 1)
)