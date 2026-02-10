SELECT
  `t1`.`bool_col` <> LAG(`t1`.`bool_col`, 1) OVER (ORDER BY `t1`.`bool_col` DESC) AS `diff_bool`
FROM (
  SELECT
    `t0`.`bool_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`