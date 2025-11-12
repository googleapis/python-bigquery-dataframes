SELECT
  ROW_NUMBER() OVER (ORDER BY NULL ASC) - 1 AS `row_number`
FROM (
  SELECT
    *
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`