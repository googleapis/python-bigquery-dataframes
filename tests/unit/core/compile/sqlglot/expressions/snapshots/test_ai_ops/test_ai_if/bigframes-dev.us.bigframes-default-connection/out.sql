SELECT
  AI.IF(
    prompt => STRUCT(
      `t0`.`string_col` AS `_field_1`,
      ' is the same as ' AS `_field_2`,
      `t0`.`string_col` AS `_field_3`
    ),
    connection_id => 'bigframes-dev.us.bigframes-default-connection'
  ) AS `result`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`