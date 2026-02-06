SELECT
  MIN(`int64_col`) OVER (
    PARTITION BY `string_col`
    ORDER BY `int64_col` DESC
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS `agg_int64`
FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`