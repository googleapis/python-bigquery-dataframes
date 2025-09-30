SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`bool_col`,
  `t0`.`int64_col` <= `t0`.`int64_col` AS `int_le_int`,
  `t0`.`int64_col` <= 1 AS `int_le_1`,
  `t0`.`int64_col` <= CAST(`t0`.`bool_col` AS INT64) AS `int_le_bool`,
  CAST(`t0`.`bool_col` AS INT64) <= `t0`.`int64_col` AS `bool_le_int`
FROM (
  SELECT
    `bool_col`,
    `int64_col`,
    `rowindex`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`