SELECT
  `t2`.`bfuid_col_561` AS `apply_on_null_true`,
  `t2`.`bfuid_col_562` AS `apply_on_null_false`
FROM (
  SELECT
    `t1`.`int64_col`,
    `my_project.my_dataset.my_routine`(`t1`.`int64_col`) AS `bfuid_col_561`,
    CASE
      WHEN `t1`.`int64_col` IS NULL
      THEN `t1`.`int64_col`
      ELSE `my_project.my_dataset.my_routine`(`t1`.`int64_col`)
    END AS `bfuid_col_562`
  FROM (
    SELECT
      `t0`.`int64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`