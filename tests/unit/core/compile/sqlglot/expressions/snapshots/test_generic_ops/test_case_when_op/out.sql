SELECT
  CASE WHEN `t0`.`bool_col` THEN `t0`.`int64_col` ELSE CAST(NULL AS INT64) END AS `single_case`,
  CASE
    WHEN `t0`.`bool_col`
    THEN `t0`.`int64_col`
    WHEN `t0`.`bool_col`
    THEN `t0`.`int64_too`
    ELSE CAST(NULL AS INT64)
  END AS `double_case`,
  CASE
    WHEN `t0`.`bool_col`
    THEN `t0`.`bool_col`
    WHEN `t0`.`bool_col`
    THEN `t0`.`bool_col`
    ELSE CAST(NULL AS BOOL)
  END AS `bool_types_case`,
  CASE
    WHEN `t0`.`bool_col`
    THEN `t0`.`int64_col`
    WHEN `t0`.`bool_col`
    THEN CAST(`t0`.`bool_col` AS INT64)
    WHEN `t0`.`bool_col`
    THEN `t0`.`float64_col`
    ELSE CAST(NULL AS FLOAT64)
  END AS `mixed_types_cast`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`