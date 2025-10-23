WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` FLOAT64, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS FLOAT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    CASE WHEN ABS(`bfcol_0`) > 1 THEN CAST('NaN' AS FLOAT64) ELSE ASIN(`bfcol_0`) END AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `float64_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST