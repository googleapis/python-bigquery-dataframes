WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64, `bfcol_2` INT64, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    GREATEST(LEAST(`bfcol_2`, `bfcol_1`), `bfcol_0`) AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_4` AS `result_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_3` ASC NULLS LAST