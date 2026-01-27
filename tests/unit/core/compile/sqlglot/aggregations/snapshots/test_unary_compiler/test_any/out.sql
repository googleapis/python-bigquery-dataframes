SELECT
  *
FROM (
  SELECT
    COALESCE(LOGICAL_OR(`t1`.`bool_col`), FALSE) AS `bool_col`,
    COALESCE(LOGICAL_OR(`t1`.`int64_col` <> 0), FALSE) AS `int64_col`
  FROM (
    SELECT
      `t0`.`bool_col`,
      `t0`.`int64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`