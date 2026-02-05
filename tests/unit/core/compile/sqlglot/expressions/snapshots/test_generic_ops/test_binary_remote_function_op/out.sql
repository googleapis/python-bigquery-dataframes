WITH `bfcte_0` AS (
  SELECT
    `float64_col`,
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  `my_project`.`my_dataset`.`my_routine`(`int64_col`, `float64_col`) AS `int64_col`
FROM `bfcte_0`