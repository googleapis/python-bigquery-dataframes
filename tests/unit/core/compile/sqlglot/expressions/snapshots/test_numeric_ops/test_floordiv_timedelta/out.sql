SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`date_col`,
  43200000000 AS `timedelta_div_numeric`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`