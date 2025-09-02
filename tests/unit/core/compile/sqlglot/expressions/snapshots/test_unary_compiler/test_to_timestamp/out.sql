SELECT
  CAST(timestamp_micros(CAST(trunc(`t0`.`int64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS `int64_col`
FROM (
  SELECT
    `int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`