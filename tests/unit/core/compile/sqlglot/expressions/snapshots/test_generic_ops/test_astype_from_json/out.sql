SELECT
  INT64(`t0`.`json_col`) AS `int64_col`,
  FLOAT64(`t0`.`json_col`) AS `float64_col`,
  BOOL(`t0`.`json_col`) AS `bool_col`,
  STRING(`t0`.`json_col`) AS `string_col`,
  SAFE.INT64(`t0`.`json_col`) AS `int64_w_safe`
FROM (
  SELECT
    `json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:50.715543')
) AS `t0`