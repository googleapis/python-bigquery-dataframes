WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    CAST(TIMESTAMP_MICROS(`bfcol_0`) AS DATETIME) AS `bfcol_2`,
    CAST(TIMESTAMP_MICROS(`bfcol_0`) AS TIME) AS `bfcol_3`,
    CAST(TIMESTAMP_MICROS(`bfcol_0`) AS TIMESTAMP) AS `bfcol_4`,
    SAFE_CAST(TIMESTAMP_MICROS(`bfcol_0`) AS TIME) AS `bfcol_5`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `int64_to_datetime`,
  `bfcol_3` AS `int64_to_time`,
  `bfcol_4` AS `int64_to_timestamp`,
  `bfcol_5` AS `int64_to_time_safe`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST