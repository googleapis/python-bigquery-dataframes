WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` GEOGRAPHY, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS GEOGRAPHY), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    ST_BUFFER(`bfcol_0`, 1.0, 8.0, FALSE) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `geography_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST