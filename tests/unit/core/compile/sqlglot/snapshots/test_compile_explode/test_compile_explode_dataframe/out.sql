WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` ARRAY<INT64>, `bfcol_2` ARRAY<STRING>, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS INT64), ARRAY<INT64>[], ARRAY<STRING>[], 0)])
), `bfcte_1` AS (
  SELECT
    *
    REPLACE (`bfcol_1`[SAFE_OFFSET(`bfcol_16`)] AS `bfcol_1`, `bfcol_2`[SAFE_OFFSET(`bfcol_16`)] AS `bfcol_2`)
  FROM `bfcte_0`
  CROSS JOIN UNNEST(GENERATE_ARRAY(0, LEAST(ARRAY_LENGTH(`bfcol_1`) - 1, ARRAY_LENGTH(`bfcol_2`) - 1))) AS `bfcol_16` WITH OFFSET AS `bfcol_9`
)
SELECT
  `bfcol_0` AS `rowindex`,
  `bfcol_0` AS `rowindex_1`,
  `bfcol_1` AS `int_list_col`,
  `bfcol_2` AS `string_list_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_3` ASC NULLS LAST,
  `bfcol_9` ASC NULLS LAST