SELECT
  ST_DISTANCE(`t1`.`geography_col`, `t1`.`geography_col`, TRUE) AS `spheroid`,
  ST_DISTANCE(`t1`.`geography_col`, `t1`.`geography_col`, FALSE) AS `no_spheroid`
FROM (
  SELECT
    `t0`.`geography_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`