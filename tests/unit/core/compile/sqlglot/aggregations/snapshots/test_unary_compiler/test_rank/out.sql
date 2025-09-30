SELECT
  (
    rank() OVER (ORDER BY `t0`.`int64_col` IS NULL ASC, `t0`.`int64_col` ASC) - 1
  ) + 1 AS `agg_int64`
FROM (
  SELECT
    `int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`