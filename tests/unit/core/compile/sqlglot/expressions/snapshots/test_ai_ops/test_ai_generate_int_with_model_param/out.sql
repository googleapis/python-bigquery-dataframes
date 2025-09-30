SELECT
  AI.GENERATE_INT(
    prompt => STRUCT(
      `t0`.`string_col` AS `_field_1`,
      ' is the same as ' AS `_field_2`,
      `t0`.`string_col` AS `_field_3`
    ),
    connection_id => 'test_connection_id',
    request_type => 'SHARED',
    model_params => JSON '{}'
  ) AS `result`
FROM (
  SELECT
    `string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
) AS `t0`