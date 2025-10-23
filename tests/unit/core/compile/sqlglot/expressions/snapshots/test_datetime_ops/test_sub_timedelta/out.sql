WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` DATE, `bfcol_1` INT64, `bfcol_2` TIMESTAMP, `bfcol_3` INT64, `bfcol_4` INT64>>[STRUCT(CAST(NULL AS DATE), CAST(NULL AS INT64), CAST(NULL AS TIMESTAMP), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_1` AS `bfcol_10`,
    `bfcol_2` AS `bfcol_11`,
    `bfcol_0` AS `bfcol_12`,
    `bfcol_3` AS `bfcol_13`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_10` AS `bfcol_19`,
    `bfcol_11` AS `bfcol_20`,
    `bfcol_13` AS `bfcol_21`,
    `bfcol_12` AS `bfcol_22`,
    TIMESTAMP_SUB(CAST(`bfcol_12` AS DATETIME), INTERVAL `bfcol_13` MICROSECOND) AS `bfcol_23`
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *,
    `bfcol_19` AS `bfcol_30`,
    `bfcol_20` AS `bfcol_31`,
    `bfcol_21` AS `bfcol_32`,
    `bfcol_22` AS `bfcol_33`,
    `bfcol_23` AS `bfcol_34`,
    TIMESTAMP_SUB(`bfcol_20`, INTERVAL `bfcol_21` MICROSECOND) AS `bfcol_35`
  FROM `bfcte_2`
), `bfcte_4` AS (
  SELECT
    *,
    `bfcol_30` AS `bfcol_43`,
    `bfcol_31` AS `bfcol_44`,
    `bfcol_32` AS `bfcol_45`,
    `bfcol_33` AS `bfcol_46`,
    `bfcol_34` AS `bfcol_47`,
    `bfcol_35` AS `bfcol_48`,
    TIMESTAMP_DIFF(CAST(`bfcol_33` AS DATETIME), CAST(`bfcol_33` AS DATETIME), MICROSECOND) AS `bfcol_49`
  FROM `bfcte_3`
), `bfcte_5` AS (
  SELECT
    *,
    `bfcol_43` AS `bfcol_58`,
    `bfcol_44` AS `bfcol_59`,
    `bfcol_45` AS `bfcol_60`,
    `bfcol_46` AS `bfcol_61`,
    `bfcol_47` AS `bfcol_62`,
    `bfcol_48` AS `bfcol_63`,
    `bfcol_49` AS `bfcol_64`,
    TIMESTAMP_DIFF(`bfcol_44`, `bfcol_44`, MICROSECOND) AS `bfcol_65`
  FROM `bfcte_4`
), `bfcte_6` AS (
  SELECT
    *,
    `bfcol_58` AS `bfcol_75`,
    `bfcol_59` AS `bfcol_76`,
    `bfcol_60` AS `bfcol_77`,
    `bfcol_61` AS `bfcol_78`,
    `bfcol_62` AS `bfcol_79`,
    `bfcol_63` AS `bfcol_80`,
    `bfcol_64` AS `bfcol_81`,
    `bfcol_65` AS `bfcol_82`,
    `bfcol_60` - `bfcol_60` AS `bfcol_83`
  FROM `bfcte_5`
)
SELECT
  `bfcol_75` AS `rowindex`,
  `bfcol_76` AS `timestamp_col`,
  `bfcol_77` AS `duration_col`,
  `bfcol_78` AS `date_col`,
  `bfcol_79` AS `date_sub_timedelta`,
  `bfcol_80` AS `timestamp_sub_timedelta`,
  `bfcol_81` AS `timestamp_sub_date`,
  `bfcol_82` AS `date_sub_timestamp`,
  `bfcol_83` AS `timedelta_sub_timedelta`
FROM `bfcte_6`
ORDER BY
  `bfcol_4` ASC NULLS LAST