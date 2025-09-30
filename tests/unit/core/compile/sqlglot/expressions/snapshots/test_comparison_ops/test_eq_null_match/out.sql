SELECT
  COALESCE(CAST(`t0`.`int64_col` AS STRING), '$NULL_SENTINEL$') = COALESCE(CAST(CAST(`t0`.`bool_col` AS INT64) AS STRING), '$NULL_SENTINEL$') AS `int64_col`
FROM (
  SELECT
    `bool_col`,
    `int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`