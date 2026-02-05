WITH `bfcte_0` AS (
  SELECT
    `geography_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  ST_BOUNDARY(`geography_col`) AS `geography_col`
FROM `bfcte_0`