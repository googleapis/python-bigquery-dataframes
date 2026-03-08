WITH `bfcte_0` AS (
  SELECT
    CORR(`int64_col`, `float64_col`) AS `bfcol_2`
  FROM (
    SELECT
      `int64_col`,
      `float64_col`
    FROM `bigframes-dev`.`sqlglot_test`.`scalar_types` AS `bft_0`
  )
)
SELECT
  `bfcol_2` AS `corr_col`
FROM `bfcte_0`