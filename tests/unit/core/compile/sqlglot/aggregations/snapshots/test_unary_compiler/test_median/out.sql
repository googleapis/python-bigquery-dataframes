SELECT
  *
FROM (
  SELECT
    APPROX_QUANTILES(`t1`.`int64_col`, 2)[offset(1)] AS `int64_col`,
    APPROX_QUANTILES(`t1`.`date_col`, 2)[offset(1)] AS `date_col`,
    APPROX_QUANTILES(`t1`.`string_col`, 2)[offset(1)] AS `string_col`
  FROM (
    SELECT
      `t0`.`date_col`,
      `t0`.`int64_col`,
      `t0`.`string_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`