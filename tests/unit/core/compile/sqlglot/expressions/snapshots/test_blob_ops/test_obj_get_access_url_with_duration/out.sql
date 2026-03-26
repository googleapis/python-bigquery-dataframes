SELECT
  `OBJ.GET_ACCESS_URL`(`t1`.`string_col`, 'READ', INTERVAL 3600 MICROSECOND) AS `string_col`
FROM (
  SELECT
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`