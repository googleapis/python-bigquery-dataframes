SELECT
  INT64(`t1`.`json_col`) AS `int64_col`,
  FLOAT64(`t1`.`json_col`) AS `float64_col`,
  BOOL(`t1`.`json_col`) AS `bool_col`,
  STRING(`t1`.`json_col`) AS `string_col`,
  SAFE.INT64(`t1`.`json_col`) AS `int64_w_safe`
FROM (
  SELECT
    `t0`.`json_col`
  FROM `bigframes-dev.sqlglot_test.json_types` AS `t0`
) AS `t1`