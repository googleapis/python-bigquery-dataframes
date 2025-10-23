WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` DATETIME, `bfcol_1` NUMERIC, `bfcol_2` FLOAT64, `bfcol_3` TIME, `bfcol_4` TIMESTAMP, `bfcol_5` INT64>>[STRUCT(
    CAST(NULL AS DATETIME),
    CAST(NULL AS NUMERIC),
    CAST(NULL AS FLOAT64),
    CAST(NULL AS TIME),
    CAST(NULL AS TIMESTAMP),
    0
  )])
), `bfcte_1` AS (
  SELECT
    *,
    UNIX_MICROS(CAST(`bfcol_0` AS TIMESTAMP)) AS `bfcol_6`,
    UNIX_MICROS(SAFE_CAST(`bfcol_0` AS TIMESTAMP)) AS `bfcol_7`,
    TIME_DIFF(CAST(`bfcol_3` AS TIME), '00:00:00', MICROSECOND) AS `bfcol_8`,
    TIME_DIFF(SAFE_CAST(`bfcol_3` AS TIME), '00:00:00', MICROSECOND) AS `bfcol_9`,
    UNIX_MICROS(`bfcol_4`) AS `bfcol_10`,
    CAST(TRUNC(`bfcol_1`) AS INT64) AS `bfcol_11`,
    CAST(TRUNC(`bfcol_2`) AS INT64) AS `bfcol_12`,
    SAFE_CAST(TRUNC(`bfcol_2`) AS INT64) AS `bfcol_13`,
    CAST('100' AS INT64) AS `bfcol_14`
  FROM `bfcte_0`
)
SELECT
  `bfcol_6` AS `datetime_col`,
  `bfcol_7` AS `datetime_w_safe`,
  `bfcol_8` AS `time_col`,
  `bfcol_9` AS `time_w_safe`,
  `bfcol_10` AS `timestamp_col`,
  `bfcol_11` AS `numeric_col`,
  `bfcol_12` AS `float64_col`,
  `bfcol_13` AS `float64_w_safe`,
  `bfcol_14` AS `str_const`
FROM `bfcte_1`
ORDER BY
  `bfcol_5` ASC NULLS LAST