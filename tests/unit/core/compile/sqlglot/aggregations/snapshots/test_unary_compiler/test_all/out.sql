WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN>>[STRUCT(CAST(NULL AS BOOLEAN))])
), `bfcte_1` AS (
  SELECT
    COALESCE(LOGICAL_AND(`bfcol_0`), TRUE) AS `bfcol_1`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `bool_col`
FROM `bfcte_1`