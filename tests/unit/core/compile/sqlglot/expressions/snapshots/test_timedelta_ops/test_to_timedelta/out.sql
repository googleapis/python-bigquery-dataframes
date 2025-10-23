WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_1` AS `bfcol_6`,
    `bfcol_0` AS `bfcol_7`,
    `bfcol_0` AS `bfcol_8`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_6` AS `bfcol_13`,
    `bfcol_7` AS `bfcol_14`,
    `bfcol_8` AS `bfcol_15`,
    `bfcol_7` * 1000000 AS `bfcol_16`
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *,
    `bfcol_13` AS `bfcol_22`,
    `bfcol_14` AS `bfcol_23`,
    `bfcol_15` AS `bfcol_24`,
    `bfcol_16` AS `bfcol_25`,
    `bfcol_14` * 604800000000 AS `bfcol_26`
  FROM `bfcte_2`
)
SELECT
  `bfcol_22` AS `rowindex`,
  `bfcol_23` AS `int64_col`,
  `bfcol_24` AS `duration_us`,
  `bfcol_25` AS `duration_s`,
  `bfcol_26` AS `duration_w`
FROM `bfcte_3`
ORDER BY
  `bfcol_2` ASC NULLS LAST