SELECT
  *
FROM (
  SELECT
    COALESCE(SUM(`t1`.`int64_col`), 0) AS `int64_col`,
    COALESCE(SUM(CAST(`t1`.`bool_col` AS INT64)), 0) AS `bool_col`
  FROM (
    SELECT
      `t0`.`int64_col`,
      `t0`.`bool_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`