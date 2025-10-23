WITH `bfcte_1` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64, `bfcol_2` STRING, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), CAST(NULL AS STRING), 0)])
), `bfcte_3` AS (
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY `bfcol_3` ASC NULLS LAST) AS `bfcol_9`
  FROM `bfcte_1`
), `bfcte_5` AS (
  SELECT
    *,
    0 AS `bfcol_15`
  FROM `bfcte_3`
), `bfcte_6` AS (
  SELECT
    `bfcol_1` AS `bfcol_16`,
    `bfcol_1` AS `bfcol_17`,
    `bfcol_0` AS `bfcol_18`,
    `bfcol_2` AS `bfcol_19`,
    `bfcol_15` AS `bfcol_20`,
    `bfcol_9` AS `bfcol_21`
  FROM `bfcte_5`
), `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_22` INT64, `bfcol_23` INT64, `bfcol_24` STRING, `bfcol_25` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), CAST(NULL AS STRING), 0)])
), `bfcte_2` AS (
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY `bfcol_25` ASC NULLS LAST) AS `bfcol_31`
  FROM `bfcte_0`
), `bfcte_4` AS (
  SELECT
    *,
    1 AS `bfcol_37`
  FROM `bfcte_2`
), `bfcte_7` AS (
  SELECT
    `bfcol_23` AS `bfcol_38`,
    `bfcol_23` AS `bfcol_39`,
    `bfcol_22` AS `bfcol_40`,
    `bfcol_24` AS `bfcol_41`,
    `bfcol_37` AS `bfcol_42`,
    `bfcol_31` AS `bfcol_43`
  FROM `bfcte_4`
), `bfcte_8` AS (
  SELECT
    *
  FROM (
    SELECT
      `bfcol_16` AS `bfcol_44`,
      `bfcol_17` AS `bfcol_45`,
      `bfcol_18` AS `bfcol_46`,
      `bfcol_19` AS `bfcol_47`,
      `bfcol_20` AS `bfcol_48`,
      `bfcol_21` AS `bfcol_49`
    FROM `bfcte_6`
    UNION ALL
    SELECT
      `bfcol_38` AS `bfcol_44`,
      `bfcol_39` AS `bfcol_45`,
      `bfcol_40` AS `bfcol_46`,
      `bfcol_41` AS `bfcol_47`,
      `bfcol_42` AS `bfcol_48`,
      `bfcol_43` AS `bfcol_49`
    FROM `bfcte_7`
  )
)
SELECT
  `bfcol_44` AS `rowindex`,
  `bfcol_45` AS `rowindex_1`,
  `bfcol_46` AS `int64_col`,
  `bfcol_47` AS `string_col`
FROM `bfcte_8`
ORDER BY
  `bfcol_48` ASC NULLS LAST,
  `bfcol_49` ASC NULLS LAST