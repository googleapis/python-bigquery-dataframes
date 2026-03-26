SELECT
  ROW_NUMBER() OVER (ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC) - 1 AS `row_number`
FROM (
  SELECT
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`