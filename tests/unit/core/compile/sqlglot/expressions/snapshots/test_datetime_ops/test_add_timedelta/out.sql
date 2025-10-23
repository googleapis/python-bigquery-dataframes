WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` DATE, `bfcol_1` INT64, `bfcol_2` TIMESTAMP, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS DATE), CAST(NULL AS INT64), CAST(NULL AS TIMESTAMP), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_1` AS `bfcol_8`,
    `bfcol_2` AS `bfcol_9`,
    `bfcol_0` AS `bfcol_10`,
    TIMESTAMP_ADD(CAST(`bfcol_0` AS DATETIME), INTERVAL 86400000000 MICROSECOND) AS `bfcol_11`
  FROM `bfcte_0`
), `bfcte_2` AS (
  SELECT
    *,
    `bfcol_8` AS `bfcol_17`,
    `bfcol_9` AS `bfcol_18`,
    `bfcol_10` AS `bfcol_19`,
    `bfcol_11` AS `bfcol_20`,
    TIMESTAMP_ADD(`bfcol_9`, INTERVAL 86400000000 MICROSECOND) AS `bfcol_21`
  FROM `bfcte_1`
), `bfcte_3` AS (
  SELECT
    *,
    `bfcol_17` AS `bfcol_28`,
    `bfcol_18` AS `bfcol_29`,
    `bfcol_19` AS `bfcol_30`,
    `bfcol_20` AS `bfcol_31`,
    `bfcol_21` AS `bfcol_32`,
    TIMESTAMP_ADD(CAST(`bfcol_19` AS DATETIME), INTERVAL 86400000000 MICROSECOND) AS `bfcol_33`
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
    TIMESTAMP_ADD(`bfcol_29`, INTERVAL 86400000000 MICROSECOND) AS `bfcol_47`
  FROM `bfcte_3`
), `bfcte_5` AS (
  SELECT
    *,
    172800000000 AS `bfcol_56`
  FROM `bfcte_4`
)
SELECT
  `bfcol_41` AS `rowindex`,
  `bfcol_42` AS `timestamp_col`,
  `bfcol_43` AS `date_col`,
  `bfcol_44` AS `date_add_timedelta`,
  `bfcol_45` AS `timestamp_add_timedelta`,
  `bfcol_46` AS `timedelta_add_date`,
  `bfcol_47` AS `timedelta_add_timestamp`,
  `bfcol_56` AS `timedelta_add_timedelta`
FROM `bfcte_5`
ORDER BY
  `bfcol_3` ASC NULLS LAST