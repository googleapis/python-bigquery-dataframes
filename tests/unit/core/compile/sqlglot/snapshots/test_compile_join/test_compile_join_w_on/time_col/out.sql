SELECT
  `t3`.`rowindex` AS `rowindex_x`,
  `t3`.`time_col`,
  `t4`.`bfuid_col_890` AS `rowindex_y`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`time_col`
  FROM (
    SELECT
      `rowindex`,
      `time_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
  ) AS `t0`
) AS `t3`
INNER JOIN (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_890`,
    `t0`.`time_col` AS `bfuid_col_891`
  FROM (
    SELECT
      `rowindex`,
      `time_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
  ) AS `t0`
) AS `t4`
  ON COALESCE(CAST(`t3`.`time_col` AS STRING), '0') = COALESCE(CAST(`t4`.`bfuid_col_891` AS STRING), '0')
  AND COALESCE(CAST(`t3`.`time_col` AS STRING), '1') = COALESCE(CAST(`t4`.`bfuid_col_891` AS STRING), '1')