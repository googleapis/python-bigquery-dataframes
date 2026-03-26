SELECT
  st_buffer(`t1`.`geography_col`, 1.0, 8.0, FALSE) AS `geography_col`
FROM (
  SELECT
    `t0`.`geography_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`