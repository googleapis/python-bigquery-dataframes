SELECT
  `t3`.`rowindex` AS `rowindex_x`,
  `t3`.`string_col`,
  `t4`.`bfuid_col_1494` AS `rowindex_y`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t3`
INNER JOIN (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1494`,
    `t0`.`string_col` AS `bfuid_col_1495`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t4`
  ON COALESCE(`t3`.`string_col`, '0') = COALESCE(`t4`.`bfuid_col_1495`, '0')
  AND COALESCE(`t3`.`string_col`, '1') = COALESCE(`t4`.`bfuid_col_1495`, '1')