WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    CAST(CAST(`bfcol_0` AS INT64) AS FLOAT64) AS `bfcol_2`,
    CAST('1.34235e4' AS FLOAT64) AS `bfcol_3`,
    SAFE_CAST(SAFE_CAST(`bfcol_0` AS INT64) AS FLOAT64) AS `bfcol_4`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `bool_col`,
  `bfcol_3` AS `str_const`,
  `bfcol_4` AS `bool_w_safe`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST