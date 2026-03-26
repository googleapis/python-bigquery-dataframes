SELECT
  CASE
    WHEN (
      `t0`.`int64_col` > 0
    ) AND (
      `t0`.`int64_col` <= 1
    )
    THEN 0
    WHEN (
      `t0`.`int64_col` > 1
    ) AND (
      `t0`.`int64_col` <= 2
    )
    THEN 1
    ELSE CAST(NULL AS INT64)
  END AS `interval_bins_labels`
FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`