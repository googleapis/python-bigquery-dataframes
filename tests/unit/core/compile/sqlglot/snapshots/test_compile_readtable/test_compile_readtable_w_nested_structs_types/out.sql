SELECT
  `t0`.`id`,
  `t0`.`id` AS `id_1`,
  `t0`.`people`
FROM (
  SELECT
    `id`,
    `people`
  FROM `bigframes-dev.sqlglot_test.nested_structs_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:51.053873')
) AS `t0`