SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  CAST(FLOOR(`t0`.`int64_col` * 1) AS INT64) AS `duration_us`,
  CAST(FLOOR(`t0`.`int64_col` * 1000000) AS INT64) AS `duration_s`,
  CAST(FLOOR(`t0`.`int64_col` * 604800000000) AS INT64) AS `duration_w`
FROM (
  SELECT
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`