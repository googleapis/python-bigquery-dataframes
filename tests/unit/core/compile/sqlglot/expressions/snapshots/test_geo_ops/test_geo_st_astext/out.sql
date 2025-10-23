WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` GEOGRAPHY, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS GEOGRAPHY), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    ST_ASTEXT(`bfcol_0`) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `geography_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST