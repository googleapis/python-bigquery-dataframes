SELECT
  *
FROM (
  SELECT
    STDDEV_SAMP(`t1`.`bfuid_col_10`) AS `int64_col`,
    STDDEV_SAMP(CAST(`t1`.`bfuid_col_11` AS INT64)) AS `bool_col`,
    CAST(FLOOR(STDDEV_SAMP(`t1`.`bfuid_col_12`)) AS INT64) AS `duration_col`,
    CAST(FLOOR(STDDEV_SAMP(`t1`.`bfuid_col_10`)) AS INT64) AS `int64_col_w_floor`
  FROM (
    SELECT
      `t0`.`int64_col` AS `bfuid_col_10`,
      `t0`.`bool_col` AS `bfuid_col_11`,
      CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64) AS `bfuid_col_12`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`