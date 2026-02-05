WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  CAST(TIMESTAMP_MICROS(`int64_col`) AS DATETIME) AS `int64_to_datetime`,
  CAST(TIMESTAMP_MICROS(`int64_col`) AS TIME) AS `int64_to_time`,
  CAST(TIMESTAMP_MICROS(`int64_col`) AS TIMESTAMP) AS `int64_to_timestamp`,
  SAFE_CAST(TIMESTAMP_MICROS(`int64_col`) AS TIME) AS `int64_to_time_safe`
FROM `bfcte_0`