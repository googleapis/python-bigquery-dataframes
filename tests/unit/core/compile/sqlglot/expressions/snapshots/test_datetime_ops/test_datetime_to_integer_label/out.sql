WITH `bfcte_0` AS (
  SELECT
    `datetime_col`,
    `timestamp_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    CAST(FLOOR(
      IEEE_DIVIDE(
        UNIX_MICROS(CAST(`datetime_col` AS TIMESTAMP)) - UNIX_MICROS(CAST(`timestamp_col` AS TIMESTAMP)),
        86400000000
      )
    ) AS INT64) AS `bfcol_2`,
    CASE
      WHEN (
        EXTRACT(YEAR FROM `datetime_col`) * 12 + EXTRACT(MONTH FROM `datetime_col`) - 1
      ) = (
        EXTRACT(YEAR FROM `timestamp_col`) * 12 + EXTRACT(MONTH FROM `timestamp_col`) - 1
      )
      THEN 0
      ELSE CAST(FLOOR(
        IEEE_DIVIDE(
          (
            EXTRACT(YEAR FROM `datetime_col`) * 12 + EXTRACT(MONTH FROM `datetime_col`) - 1
          ) - (
            EXTRACT(YEAR FROM `timestamp_col`) * 12 + EXTRACT(MONTH FROM `timestamp_col`) - 1
          ) - 1,
          1
        )
      ) AS INT64) + 1
    END AS `bfcol_3`,
    CASE
      WHEN UNIX_MICROS(
        CAST(TIMESTAMP_TRUNC(`datetime_col`, WEEK(MONDAY)) + INTERVAL 6 DAY AS TIMESTAMP)
      ) = UNIX_MICROS(
        CAST(TIMESTAMP_TRUNC(`timestamp_col`, WEEK(MONDAY)) + INTERVAL 6 DAY AS TIMESTAMP)
      )
      THEN 0
      ELSE CAST(FLOOR(
        IEEE_DIVIDE(
          UNIX_MICROS(
            CAST(TIMESTAMP_TRUNC(`datetime_col`, WEEK(MONDAY)) + INTERVAL 6 DAY AS TIMESTAMP)
          ) - UNIX_MICROS(
            CAST(TIMESTAMP_TRUNC(`timestamp_col`, WEEK(MONDAY)) + INTERVAL 6 DAY AS TIMESTAMP)
          ) - 1,
          604800000000
        )
      ) AS INT64) + 1
    END AS `bfcol_4`,
    CASE
      WHEN (
        EXTRACT(YEAR FROM `datetime_col`) * 4 + EXTRACT(QUARTER FROM `datetime_col`) - 1
      ) = (
        EXTRACT(YEAR FROM `timestamp_col`) * 4 + EXTRACT(QUARTER FROM `timestamp_col`) - 1
      )
      THEN 0
      ELSE CAST(FLOOR(
        IEEE_DIVIDE(
          (
            EXTRACT(YEAR FROM `datetime_col`) * 4 + EXTRACT(QUARTER FROM `datetime_col`) - 1
          ) - (
            EXTRACT(YEAR FROM `timestamp_col`) * 4 + EXTRACT(QUARTER FROM `timestamp_col`) - 1
          ) - 1,
          1
        )
      ) AS INT64) + 1
    END AS `bfcol_5`,
    CASE
      WHEN EXTRACT(YEAR FROM `datetime_col`) = EXTRACT(YEAR FROM `timestamp_col`)
      THEN 0
      ELSE CAST(FLOOR(
        IEEE_DIVIDE(EXTRACT(YEAR FROM `datetime_col`) - EXTRACT(YEAR FROM `timestamp_col`) - 1, 1)
      ) AS INT64) + 1
    END AS `bfcol_6`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `fixed_freq`,
  `bfcol_3` AS `non_fixed_freq_monthly`,
  `bfcol_4` AS `non_fixed_freq_weekly`,
  `bfcol_5` AS `non_fixed_freq_quarterly`,
  `bfcol_6` AS `non_fixed_freq_yearly`
FROM `bfcte_1`