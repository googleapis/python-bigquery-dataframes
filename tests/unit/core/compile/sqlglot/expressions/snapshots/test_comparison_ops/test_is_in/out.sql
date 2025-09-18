SELECT
  COALESCE(`t0`.`int64_col` IN (1, 2, 3), FALSE) AS `ints`,
  (
    `t0`.`int64_col` IS NULL
  ) OR `t0`.`int64_col` IN (123456) AS `ints_w_null`,
  COALESCE(`t0`.`int64_col` IN (1.0, 2.0, 3.0), FALSE) AS `floats`,
  COALESCE(FALSE, FALSE) AS `strings`,
  COALESCE(`t0`.`int64_col` IN (2.5, 3), FALSE) AS `mixed`,
  COALESCE(FALSE, FALSE) AS `empty`,
  COALESCE(`t0`.`int64_col` IN (123456), FALSE) AS `ints_wo_match_nulls`,
  (
    `t0`.`float64_col` IS NULL
  ) OR `t0`.`float64_col` IN (1, 2, 3) AS `float_in_ints`
FROM (
  SELECT
    `int64_col`,
    `float64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
) AS `t0`