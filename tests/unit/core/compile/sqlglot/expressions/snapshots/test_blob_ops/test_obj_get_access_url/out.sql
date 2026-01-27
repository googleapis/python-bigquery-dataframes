SELECT
  `t1`.`rowindex`,
  json_value(
    `OBJ.GET_ACCESS_URL`(
      `OBJ.MAKE_REF`(`t1`.`string_col`, 'bigframes-dev.test-region.bigframes-default-connection'),
      'R'
    ),
    '$.access_urls.read_url'
  ) AS `string_col`
FROM (
  SELECT
    `t0`.`rowindex`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`