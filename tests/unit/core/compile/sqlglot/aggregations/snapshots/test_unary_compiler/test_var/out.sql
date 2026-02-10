SELECT
  *
FROM (
  SELECT
    VARIANCE(`t1`.`int64_col`) AS `int64_col`,
    VARIANCE(CAST(`t1`.`bool_col` AS INT64)) AS `bool_col`
  FROM (
    SELECT
      `t0`.`int64_col`,
      `t0`.`bool_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`