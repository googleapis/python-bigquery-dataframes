SELECT
  *
FROM (
  SELECT
    AVG(`t1`.`bfuid_col_4`) AS `int64_col`,
    AVG(CAST(`t1`.`bfuid_col_5` AS INT64)) AS `bool_col`,
    CAST(FLOOR(AVG(`t1`.`bfuid_col_6`)) AS INT64) AS `duration_col`,
    CAST(FLOOR(AVG(`t1`.`bfuid_col_4`)) AS INT64) AS `int64_col_w_floor`
  FROM (
    SELECT
      `t0`.`int64_col` AS `bfuid_col_4`,
      `t0`.`bool_col` AS `bfuid_col_5`,
      CAST(FLOOR(`t0`.`duration_col` * 1) AS INT64) AS `bfuid_col_6`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`