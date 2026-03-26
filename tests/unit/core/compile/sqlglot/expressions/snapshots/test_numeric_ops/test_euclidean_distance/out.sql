SELECT
  `ML.DISTANCE`(`t1`.`int_list_col`, `t1`.`int_list_col`, 'EUCLIDEAN') AS `int_list_col`,
  `ML.DISTANCE`(`t1`.`numeric_list_col`, `t1`.`numeric_list_col`, 'EUCLIDEAN') AS `numeric_list_col`
FROM (
  SELECT
    `t0`.`int_list_col`,
    `t0`.`numeric_list_col`
  FROM `bigframes-dev.sqlglot_test.repeated_types` AS `t0`
) AS `t1`