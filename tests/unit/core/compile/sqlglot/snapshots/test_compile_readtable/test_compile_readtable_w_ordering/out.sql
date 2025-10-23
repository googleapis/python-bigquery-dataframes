WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
)
SELECT
  `bfcol_1` AS `rowindex`,
  `bfcol_0` AS `int64_col`
FROM `bfcte_0`
ORDER BY
  `bfcol_0` ASC NULLS LAST,
  `bfcol_2` ASC NULLS LAST