SELECT
  CAST(FLOOR(
    DATE_DIFF(
      `t1`.`date_col`,
      LAG(`t1`.`date_col`, 1) OVER (ORDER BY `t1`.`date_col` IS NULL ASC, `t1`.`date_col` ASC),
      DAY
    ) * 86400000000
  ) AS INT64) AS `diff_date`
FROM (
  SELECT
    `t0`.`date_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`