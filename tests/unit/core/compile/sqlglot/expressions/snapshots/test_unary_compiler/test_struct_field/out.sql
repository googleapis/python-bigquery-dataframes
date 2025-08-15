SELECT
  `t0`.`people`.`name` AS `people`
FROM (
  SELECT
    `people`
  FROM `bigframes-dev.sqlglot_test.nested_structs_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-15T22:12:37.907430')
) AS `t0`