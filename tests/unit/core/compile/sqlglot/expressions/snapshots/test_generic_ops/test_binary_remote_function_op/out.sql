SELECT
  `t2`.`bfuid_col_563` AS `int64_col`
FROM (
  SELECT
    `t1`.`int64_col`,
    `t1`.`float64_col`,
    `my_project.my_dataset.my_routine`(`t1`.`int64_col`, `t1`.`float64_col`) AS `bfuid_col_563`
  FROM (
    SELECT
      `t0`.`int64_col`,
      `t0`.`float64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`