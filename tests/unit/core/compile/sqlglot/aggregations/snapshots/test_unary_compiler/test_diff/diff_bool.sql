WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_0` <> LAG(`bfcol_0`, 1) OVER (ORDER BY `bfcol_0` DESC, `bfcol_1` ASC NULLS LAST) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `diff_bool`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST