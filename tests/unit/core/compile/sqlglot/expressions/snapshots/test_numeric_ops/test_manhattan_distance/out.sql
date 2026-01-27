SELECT
  `ML.DISTANCE`(`t1`.`float_list_col`, `t1`.`float_list_col`, 'MANHATTAN') AS `float_list_col`,
  `ML.DISTANCE`(`t1`.`numeric_list_col`, `t1`.`numeric_list_col`, 'MANHATTAN') AS `numeric_list_col`
FROM (
  SELECT
    `t0`.`float_list_col`,
    `t0`.`numeric_list_col`
  FROM `bigframes-dev.sqlglot_test.repeated_types` AS `t0`
) AS `t1`