SELECT
  `t3`.`rowindex` AS `rowindex_x`,
  `t3`.`time_col`,
  `t4`.`bfuid_col_1497` AS `rowindex_y`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`time_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t3`
INNER JOIN (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1497`,
    `t0`.`time_col` AS `bfuid_col_1498`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t4`
  ON COALESCE(CAST(`t3`.`time_col` AS STRING), '0') = COALESCE(CAST(`t4`.`bfuid_col_1498` AS STRING), '0')
  AND COALESCE(CAST(`t3`.`time_col` AS STRING), '1') = COALESCE(CAST(`t4`.`bfuid_col_1498` AS STRING), '1')