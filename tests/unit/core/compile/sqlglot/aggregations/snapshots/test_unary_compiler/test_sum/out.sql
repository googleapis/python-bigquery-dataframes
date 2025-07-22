SELECT
  *
FROM (
  SELECT
    COALESCE(SUM(`t1`.`int64_col`), 0) AS `int64_col_agg`
  FROM (
    SELECT
      `t0`.`int64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`