WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` INT64, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_2` AS `bfcol_8`,
    `bfcol_0` AS `bfcol_9`,
    `bfcol_1` AS `bfcol_10`,
    `bfcol_1` & `bfcol_1` AS `bfcol_11`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_8` AS `bfcol_17`,
    `bfcol_9` AS `bfcol_18`,
    `bfcol_10` AS `bfcol_19`,
    `bfcol_11` AS `bfcol_20`,
    `bfcol_9` AND `bfcol_9` AS `bfcol_21`
  FROM `bfcte_1`
)
SELECT
  `bfcol_17` AS `rowindex`,
  `bfcol_18` AS `bool_col`,
  `bfcol_19` AS `int64_col`,
  `bfcol_20` AS `int_and_int`,
  `bfcol_21` AS `bool_and_bool`
FROM `bfcte_2`
ORDER BY
  `bfcol_3` ASC NULLS LAST