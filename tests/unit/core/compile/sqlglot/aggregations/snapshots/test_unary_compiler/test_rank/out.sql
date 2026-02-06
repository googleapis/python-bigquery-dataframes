WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  RANK() OVER (ORDER BY `int64_col` DESC NULLS FIRST) AS `agg_int64`
FROM `bfcte_0`