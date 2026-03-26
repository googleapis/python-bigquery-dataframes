SELECT
  *
FROM (
  SELECT
    ARRAY_AGG(
      `t1`.`int64_col` IGNORE NULLS ORDER BY (`t1`.`int64_col` IS NULL) ASC, (`t1`.`int64_col`) ASC
    ) AS `int64_col`
  FROM (
    SELECT
      `t0`.`int64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`