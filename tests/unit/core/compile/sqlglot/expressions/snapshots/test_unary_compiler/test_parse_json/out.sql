SELECT
  `t1`.`rowindex`,
  json_value(`t1`.`string_col`, '$') AS `string_col`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`