WITH `bfcte_0` AS (
  SELECT
    `int64_col`,
    `int64_too`,
    `rowindex`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  GREATEST(LEAST(`rowindex`, `int64_too`), `int64_col`) AS `result_col`
FROM `bfcte_0`