SELECT
  `t0`.`people`.`name` AS `string`,
  `t0`.`people`.`name` AS `int`
FROM (
  SELECT
    `people`
  FROM `bigframes-dev.sqlglot_test.nested_structs_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:51.053873')
) AS `t0`