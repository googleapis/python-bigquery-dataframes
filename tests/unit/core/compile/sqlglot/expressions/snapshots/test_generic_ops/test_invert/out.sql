SELECT
  ~`t0`.`int64_col` AS `int64_col`,
  ~`t0`.`bytes_col` AS `bytes_col`,
  NOT (
    `t0`.`bool_col`
  ) AS `bool_col`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`