SELECT
  AI.GENERATE(
    prompt => STRUCT(
      `t0`.`string_col` AS `_field_1`,
      ' is the same as ' AS `_field_2`,
      `t0`.`string_col` AS `_field_3`
    ),
    endpoint => 'gemini-2.5-flash',
    request_type => 'SHARED'
  ) AS `result`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`