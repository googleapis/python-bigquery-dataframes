WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  FLOOR(`int64_col`) AS `int64_col`
FROM `bfcte_0`