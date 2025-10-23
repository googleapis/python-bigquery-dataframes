WITH `bfcte_3` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` FLOAT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS FLOAT64), 0)])
), `bfcte_7` AS (
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY `bfcol_0` ASC NULLS LAST, `bfcol_2` ASC NULLS LAST) AS `bfcol_6`
  FROM `bfcte_3`
), `bfcte_11` AS (
  SELECT
    *,
    0 AS `bfcol_10`
  FROM `bfcte_7`
), `bfcte_14` AS (
  SELECT
    `bfcol_1` AS `bfcol_11`,
    `bfcol_0` AS `bfcol_12`,
    `bfcol_10` AS `bfcol_13`,
    `bfcol_6` AS `bfcol_14`
  FROM `bfcte_11`
), `bfcte_2` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_15` BOOLEAN, `bfcol_16` INT64, `bfcol_17` FLOAT64, `bfcol_18` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), CAST(NULL AS FLOAT64), 0)])
), `bfcte_6` AS (
  SELECT
    *
  FROM `bfcte_2`
  WHERE
    `bfcol_15`
), `bfcte_10` AS (
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY `bfcol_18` ASC NULLS LAST) AS `bfcol_22`
  FROM `bfcte_6`
), `bfcte_13` AS (
  SELECT
    *,
    1 AS `bfcol_26`
  FROM `bfcte_10`
), `bfcte_15` AS (
  SELECT
    `bfcol_17` AS `bfcol_27`,
    `bfcol_16` AS `bfcol_28`,
    `bfcol_26` AS `bfcol_29`,
    `bfcol_22` AS `bfcol_30`
  FROM `bfcte_13`
), `bfcte_1` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_31` INT64, `bfcol_32` FLOAT64, `bfcol_33` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS FLOAT64), 0)])
), `bfcte_5` AS (
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY `bfcol_31` ASC NULLS LAST, `bfcol_33` ASC NULLS LAST) AS `bfcol_37`
  FROM `bfcte_1`
), `bfcte_9` AS (
  SELECT
    *,
    2 AS `bfcol_41`
  FROM `bfcte_5`
), `bfcte_16` AS (
  SELECT
    `bfcol_32` AS `bfcol_42`,
    `bfcol_31` AS `bfcol_43`,
    `bfcol_41` AS `bfcol_44`,
    `bfcol_37` AS `bfcol_45`
  FROM `bfcte_9`
), `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_46` BOOLEAN, `bfcol_47` INT64, `bfcol_48` FLOAT64, `bfcol_49` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), CAST(NULL AS FLOAT64), 0)])
), `bfcte_4` AS (
  SELECT
    *
  FROM `bfcte_0`
  WHERE
    `bfcol_46`
), `bfcte_8` AS (
  SELECT
    *,
    ROW_NUMBER() OVER (ORDER BY `bfcol_49` ASC NULLS LAST) AS `bfcol_53`
  FROM `bfcte_4`
), `bfcte_12` AS (
  SELECT
    *,
    3 AS `bfcol_57`
  FROM `bfcte_8`
), `bfcte_17` AS (
  SELECT
    `bfcol_48` AS `bfcol_58`,
    `bfcol_47` AS `bfcol_59`,
    `bfcol_57` AS `bfcol_60`,
    `bfcol_53` AS `bfcol_61`
  FROM `bfcte_12`
), `bfcte_18` AS (
  SELECT
    *
  FROM (
    SELECT
      `bfcol_11` AS `bfcol_62`,
      `bfcol_12` AS `bfcol_63`,
      `bfcol_13` AS `bfcol_64`,
      `bfcol_14` AS `bfcol_65`
    FROM `bfcte_14`
    UNION ALL
    SELECT
      `bfcol_27` AS `bfcol_62`,
      `bfcol_28` AS `bfcol_63`,
      `bfcol_29` AS `bfcol_64`,
      `bfcol_30` AS `bfcol_65`
    FROM `bfcte_15`
    UNION ALL
    SELECT
      `bfcol_42` AS `bfcol_62`,
      `bfcol_43` AS `bfcol_63`,
      `bfcol_44` AS `bfcol_64`,
      `bfcol_45` AS `bfcol_65`
    FROM `bfcte_16`
    UNION ALL
    SELECT
      `bfcol_58` AS `bfcol_62`,
      `bfcol_59` AS `bfcol_63`,
      `bfcol_60` AS `bfcol_64`,
      `bfcol_61` AS `bfcol_65`
    FROM `bfcte_17`
  )
)
SELECT
  `bfcol_62` AS `float64_col`,
  `bfcol_63` AS `int64_col`
FROM `bfcte_18`
ORDER BY
  `bfcol_64` ASC NULLS LAST,
  `bfcol_65` ASC NULLS LAST