SELECT
  `t1`.`rowindex`,
  `OBJ.FETCH_METADATA`(
    `OBJ.MAKE_REF`(`t1`.`string_col`, 'bigframes-dev.test-region.bigframes-default-connection')
  ).`version` AS `version`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`