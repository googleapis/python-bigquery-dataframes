SELECT
  `t0`.`rowindex`,
  `t0`.`int64_col`,
  `t0`.`string_col`
FROM (
  SELECT
    `rowindex`,
    `int64_col`,
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types`
  WHERE
    `rowindex` > 0 AND `string_col` IN ('Hello, World!')
) AS `t0`