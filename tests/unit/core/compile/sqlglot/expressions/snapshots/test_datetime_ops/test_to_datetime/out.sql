SELECT
  CAST(CAST(timestamp_micros(CAST(trunc(`t0`.`int64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS DATETIME) AS `int64_col`
FROM (
  SELECT
    `int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`