SELECT
  `t0`.`rowindex`,
  `t0`.`bool_col`,
  `t0`.`int64_col`,
  `t0`.`int64_col` ^ `t0`.`int64_col` AS `int_and_int`,
  (
    `t0`.`bool_col` AND NOT `t0`.`bool_col`
  )
  OR (
    NOT `t0`.`bool_col` AND `t0`.`bool_col`
  ) AS `bool_and_bool`,
  (
    `t0`.`bool_col` AND NOT CAST(NULL AS BOOL)
  )
  OR (
    NOT `t0`.`bool_col` AND CAST(NULL AS BOOL)
  ) AS `bool_and_null`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`