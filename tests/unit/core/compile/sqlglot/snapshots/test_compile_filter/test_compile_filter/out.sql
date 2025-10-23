WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_1` AS `bfcol_7`,
    `bfcol_1` AS `bfcol_8`,
    `bfcol_0` AS `bfcol_9`,
    `bfcol_1` >= 1 AS `bfcol_10`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *
  FROM `bfcte_1`
  WHERE
    `bfcol_10`
)
SELECT
  `bfcol_7` AS `rowindex`,
  `bfcol_8` AS `rowindex_1`,
  `bfcol_9` AS `int64_col`
FROM `bfcte_2`
ORDER BY
  `bfcol_2` ASC NULLS LAST