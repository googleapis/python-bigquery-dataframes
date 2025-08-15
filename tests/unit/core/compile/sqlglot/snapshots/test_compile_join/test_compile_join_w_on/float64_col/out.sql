SELECT
  `t3`.`rowindex` AS `rowindex_x`,
  `t3`.`float64_col`,
  `t4`.`bfuid_col_368` AS `rowindex_y`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`float64_col`
  FROM (
    SELECT
      `float64_col`,
      `rowindex`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
  ) AS `t0`
) AS `t3`
INNER JOIN (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_368`,
    `t0`.`float64_col` AS `bfuid_col_369`
  FROM (
    SELECT
      `float64_col`,
      `rowindex`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:35.305875')
  ) AS `t0`
) AS `t4`
  ON IF(IS_NAN(`t3`.`float64_col`), 2, COALESCE(`t3`.`float64_col`, 0)) = IF(IS_NAN(`t4`.`bfuid_col_369`), 2, COALESCE(`t4`.`bfuid_col_369`, 0))
  AND IF(IS_NAN(`t3`.`float64_col`), 3, COALESCE(`t3`.`float64_col`, 1)) = IF(IS_NAN(`t4`.`bfuid_col_369`), 3, COALESCE(`t4`.`bfuid_col_369`, 1))