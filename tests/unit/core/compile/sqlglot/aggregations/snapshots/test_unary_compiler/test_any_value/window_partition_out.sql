SELECT
  ANY_VALUE(`int64_col`) OVER (
    PARTITION BY `string_col`
    ORDER BY `int64_col` ASC NULLS LAST
    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS `agg_int64`
FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`