SELECT
  *
FROM (
  SELECT
    COUNT(1) AS `size`
  FROM (
    SELECT
      `t0`.`rowindex` AS `bfuid_col_1`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`