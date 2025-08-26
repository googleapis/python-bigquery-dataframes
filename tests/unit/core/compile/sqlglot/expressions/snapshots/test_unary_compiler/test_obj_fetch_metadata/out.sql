SELECT
  `t0`.`rowindex`,
  `OBJ.FETCH_METADATA`(
    `OBJ.MAKE_REF`(`t0`.`string_col`, 'bigframes-dev.test-region.bigframes-default-connection')
  ).`version` AS `version`
FROM (
  SELECT
    `rowindex`,
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:49:28.159676')
) AS `t0`