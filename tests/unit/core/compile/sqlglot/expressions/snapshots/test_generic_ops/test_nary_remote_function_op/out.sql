SELECT
  `t2`.`bfuid_col_564` AS `int64_col`
FROM (
  SELECT
    `t1`.`int64_col`,
    `t1`.`float64_col`,
    `t1`.`string_col`,
    `my_project.my_dataset.my_routine`(`t1`.`int64_col`, `t1`.`float64_col`, `t1`.`string_col`) AS `bfuid_col_564`
  FROM (
    SELECT
      `t0`.`int64_col`,
      `t0`.`float64_col`,
      `t0`.`string_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`