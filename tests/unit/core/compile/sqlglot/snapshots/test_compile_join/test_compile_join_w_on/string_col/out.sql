SELECT
  `t3`.`rowindex` AS `rowindex_x`,
  `t3`.`string_col`,
  `t4`.`bfuid_col_799` AS `rowindex_y`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`string_col`
  FROM (
    SELECT
      `rowindex`,
      `string_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
  ) AS `t0`
) AS `t3`
INNER JOIN (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_799`,
    `t0`.`string_col` AS `bfuid_col_800`
  FROM (
    SELECT
      `rowindex`,
      `string_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
  ) AS `t0`
) AS `t4`
  ON COALESCE(`t3`.`string_col`, '0') = COALESCE(`t4`.`bfuid_col_800`, '0')
  AND COALESCE(`t3`.`string_col`, '1') = COALESCE(`t4`.`bfuid_col_800`, '1')