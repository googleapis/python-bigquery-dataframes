SELECT
  `t0`.`rowindex`,
  json_value(
    `OBJ.GET_ACCESS_URL`(
      `OBJ.MAKE_REF`(`t0`.`string_col`, 'bigframes-dev.test-region.bigframes-default-connection'),
      'R'
    ),
    '$.access_urls.read_url'
  ) AS `string_col`
FROM (
  SELECT
    `rowindex`,
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`