WITH `bfcte_0` AS (
  SELECT
    `timestamp_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  EXTRACT(YEAR FROM `timestamp_col`) AS `timestamp_col`
FROM `bfcte_0`