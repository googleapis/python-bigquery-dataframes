WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` FLOAT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS FLOAT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_0` AS `bfcol_3`,
    `bfcol_1` <> 0 AS `bfcol_4`,
    `bfcol_1` <> 0 AS `bfcol_5`
  FROM `bfcte_0`
)
SELECT
  `bfcol_3` AS `bool_col`,
  `bfcol_4` AS `float64_col`,
  `bfcol_5` AS `float64_w_safe`
FROM `bfcte_1`
ORDER BY
  `bfcol_2` ASC NULLS LAST