SELECT
  CASE
    WHEN (
      `t0`.`int64_col` > 0
    ) AND (
      `t0`.`int64_col` <= 1
    )
    THEN STRUCT(0 AS `left_exclusive`, 1 AS `right_inclusive`)
    WHEN (
      `t0`.`int64_col` > 1
    ) AND (
      `t0`.`int64_col` <= 2
    )
    THEN STRUCT(1 AS `left_exclusive`, 2 AS `right_inclusive`)
    ELSE CAST(NULL AS STRUCT<`left_exclusive` INT64, `right_inclusive` INT64>)
  END AS `interval_bins`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`