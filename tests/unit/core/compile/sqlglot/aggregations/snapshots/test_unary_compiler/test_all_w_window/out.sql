SELECT
  COALESCE(
    LOGICAL_AND(`bool_col`) OVER (
      ORDER BY `bool_col` ASC NULLS LAST
      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ),
    TRUE
  ) AS `agg_bool`
FROM `bigframes-dev`.`sqlglot_test`.`scalar_types`