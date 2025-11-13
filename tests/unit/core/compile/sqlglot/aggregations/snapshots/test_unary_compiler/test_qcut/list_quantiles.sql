WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
), `bfcte_1` AS (
  SELECT
    *,
    CASE
      WHEN PERCENT_RANK() OVER () < 0
      THEN NULL
      WHEN PERCENT_RANK() OVER () <= 0.25
      THEN 0
      WHEN PERCENT_RANK() OVER () <= 0.5
      THEN 1
      WHEN PERCENT_RANK() OVER () <= 0.75
      THEN 2
      WHEN PERCENT_RANK() OVER () <= 1
      THEN 3
      ELSE NULL
    END AS `bfcol_1`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `list_quantiles`
FROM `bfcte_1`