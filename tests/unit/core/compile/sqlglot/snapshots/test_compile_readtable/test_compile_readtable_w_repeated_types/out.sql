SELECT
  `t0`.`rowindex`,
  `t0`.`rowindex` AS `rowindex_1`,
  `t0`.`int_list_col`,
  `t0`.`bool_list_col`,
  `t0`.`float_list_col`,
  `t0`.`date_list_col`,
  `t0`.`date_time_list_col`,
  `t0`.`numeric_list_col`,
  `t0`.`string_list_col`
FROM (
  SELECT
    `rowindex`,
    `int_list_col`,
    `bool_list_col`,
    `float_list_col`,
    `date_list_col`,
    `date_time_list_col`,
    `numeric_list_col`,
    `string_list_col`
  FROM `bigframes-dev.sqlglot_test.repeated_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:30.548576')
) AS `t0`