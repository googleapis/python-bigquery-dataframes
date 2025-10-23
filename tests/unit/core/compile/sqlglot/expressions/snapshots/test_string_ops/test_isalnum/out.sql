WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` STRING, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    REGEXP_CONTAINS(`bfcol_0`, '^(\\p{N}|\\p{L})+$') AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `string_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST