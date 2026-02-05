WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  `my_project`.`my_dataset`.`my_routine`(`int64_col`) AS `apply_on_null_true`,
  IF(
    `int64_col` IS NULL,
    `int64_col`,
    `my_project`.`my_dataset`.`my_routine`(`int64_col`)
  ) AS `apply_on_null_false`
FROM `bfcte_0`