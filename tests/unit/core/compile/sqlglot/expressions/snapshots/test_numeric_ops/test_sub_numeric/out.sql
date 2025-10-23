WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` INT64, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_2` AS `bfcol_8`,
    `bfcol_1` AS `bfcol_9`,
    `bfcol_0` AS `bfcol_10`,
    `bfcol_1` - `bfcol_1` AS `bfcol_11`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_8` AS `bfcol_17`,
    `bfcol_9` AS `bfcol_18`,
    `bfcol_10` AS `bfcol_19`,
    `bfcol_11` AS `bfcol_20`,
    `bfcol_9` - 1 AS `bfcol_21`
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *,
    `bfcol_17` AS `bfcol_28`,
    `bfcol_18` AS `bfcol_29`,
    `bfcol_19` AS `bfcol_30`,
    `bfcol_20` AS `bfcol_31`,
    `bfcol_21` AS `bfcol_32`,
    `bfcol_18` - CAST(`bfcol_19` AS INT64) AS `bfcol_33`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    *,
    `bfcol_28` AS `bfcol_41`,
    `bfcol_29` AS `bfcol_42`,
    `bfcol_30` AS `bfcol_43`,
    `bfcol_31` AS `bfcol_44`,
    `bfcol_32` AS `bfcol_45`,
    `bfcol_33` AS `bfcol_46`,
    CAST(`bfcol_30` AS INT64) - `bfcol_29` AS `bfcol_47`
  FROM `bfcte_3`
)
SELECT
  `bfcol_41` AS `rowindex`,
  `bfcol_42` AS `int64_col`,
  `bfcol_43` AS `bool_col`,
  `bfcol_44` AS `int_add_int`,
  `bfcol_45` AS `int_add_1`,
  `bfcol_46` AS `int_add_bool`,
  `bfcol_47` AS `bool_add_int`
FROM `bfcte_4`
ORDER BY
  `bfcol_3` ASC NULLS LAST