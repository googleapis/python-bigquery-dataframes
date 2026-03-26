SELECT
  *
FROM (
  SELECT
    PERCENTILE_CONT(`t1`.`int64_col`, 0.5) AS `int64`,
    PERCENTILE_CONT(CAST(`t1`.`bool_col` AS INT64), 0.5) AS `bool`,
    CAST(FLOOR(PERCENTILE_CONT(`t1`.`int64_col`, 0.5)) AS INT64) AS `int64_w_floor`
  FROM (
    SELECT
      `t0`.`int64_col`,
      `t0`.`bool_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`