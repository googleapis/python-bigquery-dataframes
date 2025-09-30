SELECT
  `t0`.`rowindex`,
  `t0`.`bool_col`,
  `t0`.`bytes_col`,
  `t0`.`date_col`,
  `t0`.`datetime_col`,
  `t0`.`geography_col`,
  `t0`.`int64_col`,
  `t0`.`int64_too`,
  `t0`.`numeric_col`,
  `t0`.`float64_col`,
  `t0`.`rowindex` AS `rowindex_1`,
  `t0`.`rowindex_2`,
  `t0`.`string_col`,
  `t0`.`time_col`,
  `t0`.`timestamp_col`,
  `t0`.`duration_col`
FROM (
  SELECT
    `bool_col`,
    `bytes_col`,
    `date_col`,
    `datetime_col`,
    `geography_col`,
    `int64_col`,
    `int64_too`,
    `numeric_col`,
    `float64_col`,
    `rowindex`,
    `rowindex_2`,
    `string_col`,
    `time_col`,
    `timestamp_col`,
    `duration_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`