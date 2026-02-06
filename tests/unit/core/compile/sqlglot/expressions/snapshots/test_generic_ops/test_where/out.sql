WITH `bfcte_0` AS (
  SELECT
    `bool_col`,
    `float64_col`,
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  IF(`bool_col`, `int64_col`, `float64_col`) AS `result_col`
FROM `bfcte_0`