SELECT
  `t3`.`int64_col`,
  `t4`.`int64_too`
FROM (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1`,
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t3`
LEFT OUTER JOIN (
  SELECT
    `t0`.`int64_col` AS `bfuid_col_1482`,
    `t0`.`int64_too`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t4`
  ON COALESCE(`t3`.`bfuid_col_1`, 0) = COALESCE(`t4`.`bfuid_col_1482`, 0)
  AND COALESCE(`t3`.`bfuid_col_1`, 1) = COALESCE(`t4`.`bfuid_col_1482`, 1)