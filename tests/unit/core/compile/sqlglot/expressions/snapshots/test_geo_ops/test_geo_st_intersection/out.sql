WITH `bfcte_0` AS (
  SELECT
    `geography_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  ST_INTERSECTION(`geography_col`, `geography_col`) AS `geography_col`
FROM `bfcte_0`