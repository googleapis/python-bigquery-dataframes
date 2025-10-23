WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` FLOAT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS FLOAT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    COALESCE(`bfcol_0` IN (1, 2, 3), FALSE) AS `bfcol_3`,
    (
      `bfcol_0` IS NULL
    ) OR `bfcol_0` IN (123456) AS `bfcol_4`,
    COALESCE(`bfcol_0` IN (1.0, 2.0, 3.0), FALSE) AS `bfcol_5`,
    FALSE AS `bfcol_6`,
    COALESCE(`bfcol_0` IN (2.5, 3), FALSE) AS `bfcol_7`,
    FALSE AS `bfcol_8`,
    COALESCE(`bfcol_0` IN (123456), FALSE) AS `bfcol_9`,
    (
      `bfcol_1` IS NULL
    ) OR `bfcol_1` IN (1, 2, 3) AS `bfcol_10`
  FROM `bfcte_0`
)
SELECT
  `bfcol_3` AS `ints`,
  `bfcol_4` AS `ints_w_null`,
  `bfcol_5` AS `floats`,
  `bfcol_6` AS `strings`,
  `bfcol_7` AS `mixed`,
  `bfcol_8` AS `empty`,
  `bfcol_9` AS `ints_wo_match_nulls`,
  `bfcol_10` AS `float_in_ints`
FROM `bfcte_1`
ORDER BY
  `bfcol_2` ASC NULLS LAST