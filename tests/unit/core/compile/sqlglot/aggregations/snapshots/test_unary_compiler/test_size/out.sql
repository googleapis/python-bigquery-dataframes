SELECT
  *
FROM (
  SELECT
    COUNT(1) AS `string_col_agg`
  FROM (
    SELECT
      `t0`.`string_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`