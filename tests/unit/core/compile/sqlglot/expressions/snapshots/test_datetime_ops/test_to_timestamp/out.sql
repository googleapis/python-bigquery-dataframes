SELECT
  CAST(timestamp_micros(CAST(trunc(`t1`.`int64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS `int64_col`,
  CAST(timestamp_micros(CAST(trunc(`t1`.`float64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS `float64_col`,
  CAST(timestamp_micros(`t1`.`int64_col` * 1000000) AS TIMESTAMP) AS `int64_col_s`,
  CAST(timestamp_micros(`t1`.`int64_col` * 1000) AS TIMESTAMP) AS `int64_col_ms`,
  CAST(timestamp_micros(`t1`.`int64_col` * 1) AS TIMESTAMP) AS `int64_col_us`,
  CAST(timestamp_micros(CAST(trunc(`t1`.`int64_col` * 0.001) AS INT64)) AS TIMESTAMP) AS `int64_col_ns`,
  TIMESTAMP(`t1`.`datetime_col`) AS `datetime_col`,
  parse_timestamp('%Y-%m-%d', `t1`.`string_col`, 'UTC') AS `string_col_fmt`
FROM (
  SELECT
    `t0`.`datetime_col`,
    `t0`.`int64_col`,
    `t0`.`float64_col`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`