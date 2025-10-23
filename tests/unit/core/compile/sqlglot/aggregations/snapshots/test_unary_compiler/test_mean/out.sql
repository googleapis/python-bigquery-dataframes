WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), CAST(NULL AS INT64))])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_1` AS `bfcol_6`,
    `bfcol_0` AS `bfcol_7`,
    `bfcol_2` AS `bfcol_8`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    AVG(`bfcol_6`) AS `bfcol_12`,
    AVG(CAST(`bfcol_7` AS INT64)) AS `bfcol_13`,
    CAST(FLOOR(AVG(`bfcol_8`)) AS INT64) AS `bfcol_14`,
    CAST(FLOOR(AVG(`bfcol_6`)) AS INT64) AS `bfcol_15`
  FROM `bfcte_1`
)
SELECT
  `bfcol_12` AS `int64_col`,
  `bfcol_13` AS `bool_col`,
  `bfcol_14` AS `duration_col`,
  `bfcol_15` AS `int64_col_w_floor`
FROM `bfcte_2`