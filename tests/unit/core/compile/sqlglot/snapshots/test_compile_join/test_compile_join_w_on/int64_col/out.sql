SELECT
  `t3`.`rowindex` AS `rowindex_x`,
  `t3`.`int64_col`,
  `t4`.`bfuid_col_881` AS `rowindex_y`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`int64_col`
  FROM (
    SELECT
      `int64_col`,
      `rowindex`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
  ) AS `t0`
) AS `t3`
INNER JOIN (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_881`,
    `t0`.`int64_col` AS `bfuid_col_882`
  FROM (
    SELECT
      `int64_col`,
      `rowindex`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
  ) AS `t0`
) AS `t4`
  ON COALESCE(`t3`.`int64_col`, 0) = COALESCE(`t4`.`bfuid_col_882`, 0)
  AND COALESCE(`t3`.`int64_col`, 1) = COALESCE(`t4`.`bfuid_col_882`, 1)