WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` FLOAT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS FLOAT64))])
), `bfcte_1` AS (
  SELECT
    CORR(`bfcol_0`, `bfcol_1`) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `corr_col`
FROM `bfcte_1`