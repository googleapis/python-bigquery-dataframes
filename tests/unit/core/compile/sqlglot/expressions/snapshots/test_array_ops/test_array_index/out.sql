WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` ARRAY<STRING>, `bfcol_1` INT64>>[STRUCT(ARRAY<STRING>[], 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_0`[SAFE_OFFSET(1)] AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `string_list_col`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST