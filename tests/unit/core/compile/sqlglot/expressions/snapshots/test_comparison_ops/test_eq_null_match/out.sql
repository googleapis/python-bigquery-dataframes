WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    COALESCE(CAST(`bfcol_1` AS STRING), '$NULL_SENTINEL$') = COALESCE(CAST(CAST(`bfcol_0` AS INT64) AS STRING), '$NULL_SENTINEL$') AS `bfcol_6`
  FROM `bfcte_0`
)
SELECT
  `bfcol_6` AS `int64_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_2` ASC NULLS LAST