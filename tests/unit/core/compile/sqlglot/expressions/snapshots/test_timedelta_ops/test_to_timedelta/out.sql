SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`float64_col`,
  CAST(FLOOR(`t0`.`int64_col` * 1) AS INT64) AS `duration_us`,
  CAST(FLOOR(`t0`.`float64_col` * 1000000) AS INT64) AS `duration_s`,
  CAST(FLOOR(`t0`.`int64_col` * 3600000000) AS INT64) AS `duration_w`,
  CAST(FLOOR(`t0`.`int64_col` * 1) AS INT64) AS `duration_on_duration`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`