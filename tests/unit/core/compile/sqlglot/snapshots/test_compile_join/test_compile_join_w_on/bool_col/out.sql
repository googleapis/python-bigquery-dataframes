SELECT
  `t3`.`rowindex` AS `rowindex_x`,
  `t3`.`bool_col`,
  `t4`.`bfuid_col_1485` AS `rowindex_y`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`bool_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t3`
INNER JOIN (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1485`,
    `t0`.`bool_col` AS `bfuid_col_1486`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t4`
  ON COALESCE(CAST(`t3`.`bool_col` AS STRING), '0') = COALESCE(CAST(`t4`.`bfuid_col_1486` AS STRING), '0')
  AND COALESCE(CAST(`t3`.`bool_col` AS STRING), '1') = COALESCE(CAST(`t4`.`bfuid_col_1486` AS STRING), '1')