WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` BYTES, `bfcol_2` DATE, `bfcol_3` DATETIME, `bfcol_4` GEOGRAPHY, `bfcol_5` INT64, `bfcol_6` INT64, `bfcol_7` NUMERIC, `bfcol_8` FLOAT64, `bfcol_9` INT64, `bfcol_10` INT64, `bfcol_11` STRING, `bfcol_12` TIME, `bfcol_13` TIMESTAMP, `bfcol_14` INT64, `bfcol_15` INT64>>[STRUCT(
    CAST(NULL AS BOOLEAN),
    CAST(NULL AS BYTES),
    CAST(NULL AS DATE),
    CAST(NULL AS DATETIME),
    CAST(NULL AS GEOGRAPHY),
    CAST(NULL AS INT64),
    CAST(NULL AS INT64),
    CAST(NULL AS NUMERIC),
    CAST(NULL AS FLOAT64),
    CAST(NULL AS INT64),
    CAST(NULL AS INT64),
    CAST(NULL AS STRING),
    CAST(NULL AS TIME),
    CAST(NULL AS TIMESTAMP),
    CAST(NULL AS INT64),
    0
  )])
)
SELECT
  `bfcol_9` AS `rowindex`,
  `bfcol_0` AS `bool_col`,
  `bfcol_1` AS `bytes_col`,
  `bfcol_2` AS `date_col`,
  `bfcol_3` AS `datetime_col`,
  `bfcol_4` AS `geography_col`,
  `bfcol_5` AS `int64_col`,
  `bfcol_6` AS `int64_too`,
  `bfcol_7` AS `numeric_col`,
  `bfcol_8` AS `float64_col`,
  `bfcol_9` AS `rowindex_1`,
  `bfcol_10` AS `rowindex_2`,
  `bfcol_11` AS `string_col`,
  `bfcol_12` AS `time_col`,
  `bfcol_13` AS `timestamp_col`,
  `bfcol_14` AS `duration_col`
FROM `bfcte_0`
ORDER BY
  `bfcol_15` ASC NULLS LAST