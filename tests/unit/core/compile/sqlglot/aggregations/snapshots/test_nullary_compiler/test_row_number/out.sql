WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY `bfcol_1` ASC NULLS LAST) AS `bfcol_3`
  FROM `bfcte_0`
)
SELECT
  `bfcol_3` AS `row_number`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST