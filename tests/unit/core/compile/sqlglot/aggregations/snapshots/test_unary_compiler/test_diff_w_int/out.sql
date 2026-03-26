SELECT
  `t1`.`int64_col` - LAG(`t1`.`int64_col`, 1) OVER (ORDER BY `t1`.`int64_col` IS NULL ASC, `t1`.`int64_col` ASC) AS `diff_int`
FROM (
  SELECT
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`