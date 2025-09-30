SELECT
  `t4`.`int64_col`,
  `t5`.`int64_too`
FROM (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1`,
    `t0`.`int64_col`
  FROM (
    SELECT
      `int64_col`,
      `rowindex`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
  ) AS `t0`
) AS `t4`
LEFT OUTER JOIN (
  SELECT
    `t1`.`int64_col` AS `bfuid_col_875`,
    `t1`.`int64_too`
  FROM (
    SELECT
      `int64_col`,
      `int64_too`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
  ) AS `t1`
) AS `t5`
  ON COALESCE(`t4`.`bfuid_col_1`, 0) = COALESCE(`t5`.`bfuid_col_875`, 0)
  AND COALESCE(`t4`.`bfuid_col_1`, 1) = COALESCE(`t5`.`bfuid_col_875`, 1)