WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64))])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_1` AS `bfcol_2`,
    `bfcol_0` AS `bfcol_3`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    `bfcol_3`,
    COALESCE(SUM(`bfcol_2`), 0) AS `bfcol_6`
  FROM `bfcte_1`
  GROUP BY
    `bfcol_3`
)
SELECT
  `bfcol_3` AS `bool_col`,
  `bfcol_6` AS `int64_too`
FROM `bfcte_2`
ORDER BY
  `bfcol_3` ASC NULLS LAST