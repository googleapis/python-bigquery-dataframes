SELECT
  `t0`.`rowindex`,
  `t0`.`timestamp_col`,
  `t0`.`date_col`,
  43200000000 AS `timedelta_div_numeric`
FROM (
  SELECT
    `date_col`,
    `rowindex`,
    `timestamp_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`