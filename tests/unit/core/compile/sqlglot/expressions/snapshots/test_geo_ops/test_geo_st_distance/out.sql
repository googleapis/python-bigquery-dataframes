WITH `bfcte_0` AS (
  SELECT
    `geography_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  ST_DISTANCE(`geography_col`, `geography_col`, TRUE) AS `spheroid`,
  ST_DISTANCE(`geography_col`, `geography_col`, FALSE) AS `no_spheroid`
FROM `bfcte_0`