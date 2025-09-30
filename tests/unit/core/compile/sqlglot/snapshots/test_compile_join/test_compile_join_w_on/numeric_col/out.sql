SELECT
  `t3`.`rowindex` AS `rowindex_x`,
  `t3`.`numeric_col`,
  `t4`.`bfuid_col_893` AS `rowindex_y`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`numeric_col`
  FROM (
    SELECT
      `numeric_col`,
      `rowindex`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
  ) AS `t0`
) AS `t3`
INNER JOIN (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_893`,
    `t0`.`numeric_col` AS `bfuid_col_894`
  FROM (
    SELECT
      `numeric_col`,
      `rowindex`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
  ) AS `t0`
) AS `t4`
  ON COALESCE(`t3`.`numeric_col`, 0) = COALESCE(`t4`.`bfuid_col_894`, 0)
  AND COALESCE(`t3`.`numeric_col`, 1) = COALESCE(`t4`.`bfuid_col_894`, 1)