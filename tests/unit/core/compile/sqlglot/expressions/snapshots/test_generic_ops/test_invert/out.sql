WITH `bfcte_0` AS (
  SELECT
    `bool_col`,
    `bytes_col`,
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  ~(
    `int64_col`
  ) AS `int64_col`,
  ~(
    `bytes_col`
  ) AS `bytes_col`,
  NOT (
    `bool_col`
  ) AS `bool_col`
FROM `bfcte_0`