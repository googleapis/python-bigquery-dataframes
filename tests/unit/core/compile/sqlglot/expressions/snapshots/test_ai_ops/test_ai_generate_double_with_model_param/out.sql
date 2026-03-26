SELECT
  AI.GENERATE_DOUBLE(
    prompt => STRUCT(
      `t0`.`string_col` AS `_field_1`,
      ' is the same as ' AS `_field_2`,
      `t0`.`string_col` AS `_field_3`
    ),
    request_type => 'SHARED',
    model_params => JSON '{}'
  ) AS `result`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`