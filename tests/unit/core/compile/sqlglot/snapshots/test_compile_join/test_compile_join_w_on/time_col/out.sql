WITH `bfcte_1` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` TIME>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS TIME))])
), `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_2` INT64, `bfcol_3` TIME>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS TIME))])
), `bfcte_2` AS (
  SELECT
    `bfcol_2` AS `bfcol_4`,
    `bfcol_3` AS `bfcol_5`
  FROM `bfcte_0`
), `bfcte_3` AS (
  SELECT
    *
  FROM `bfcte_1`
  INNER JOIN `bfcte_2`
    ON COALESCE(CAST(`bfcol_1` AS STRING), '0') = COALESCE(CAST(`bfcol_5` AS STRING), '0')
    AND COALESCE(CAST(`bfcol_1` AS STRING), '1') = COALESCE(CAST(`bfcol_5` AS STRING), '1')
)
SELECT
  `bfcol_0` AS `rowindex_x`,
  `bfcol_1` AS `time_col`,
  `bfcol_4` AS `rowindex_y`
FROM `bfcte_3`