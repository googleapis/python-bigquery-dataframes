WITH `bfcte_0` AS (
  SELECT
    `int64_col`
  FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`
)
SELECT
  *,
  VARIANCE(`int64_col`) OVER (
    ORDER BY `int64_col` DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS `agg_int64`
FROM `bfcte_0`