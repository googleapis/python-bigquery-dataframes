WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    LAG(`bfcol_0`, 1) OVER (ORDER BY `bfcol_0` ASC, `bfcol_1` ASC NULLS LAST) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `lag`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST