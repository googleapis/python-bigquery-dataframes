SELECT
  *
FROM (
  SELECT
    COALESCE(LOGICAL_AND(`t1`.`bool_col`), TRUE) AS `bool_col`
  FROM (
    SELECT
      `t0`.`bool_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`