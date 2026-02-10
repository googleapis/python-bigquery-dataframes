SELECT
  *
FROM (
  SELECT
    COUNT(`t1`.`int64_col`) AS `int64_col`
  FROM (
    SELECT
      `t0`.`int64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`