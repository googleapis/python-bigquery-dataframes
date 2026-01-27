SELECT
  st_length(`t1`.`geography_col`, TRUE) AS `geography_col`
FROM (
  SELECT
    `t0`.`geography_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`