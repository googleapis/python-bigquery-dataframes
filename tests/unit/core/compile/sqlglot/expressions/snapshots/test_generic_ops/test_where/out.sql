WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` FLOAT64, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), CAST(NULL AS FLOAT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    IF(`bfcol_0`, `bfcol_1`, `bfcol_2`) AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_4` AS `result_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_3` ASC NULLS LAST