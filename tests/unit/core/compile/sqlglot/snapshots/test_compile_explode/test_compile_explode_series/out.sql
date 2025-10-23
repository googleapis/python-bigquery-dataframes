WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` ARRAY<INT64>, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), ARRAY<INT64>[], 0)])
), `bfcte_1` AS (
  SELECT
    *
    REPLACE (`bfcol_11` AS `bfcol_1`)
  FROM `bfcte_0`
  CROSS JOIN UNNEST(`bfcol_1`) AS `bfcol_11` WITH OFFSET AS `bfcol_6`
)
SELECT
  `bfcol_0` AS `rowindex`,
  `bfcol_1` AS `int_list_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_2` ASC NULLS LAST,
  `bfcol_6` ASC NULLS LAST