SELECT
  *
FROM (
  SELECT
    APPROX_QUANTILES(`t1`.`int64_col`, 4)[safe_offset(1)] AS `q1`,
    APPROX_QUANTILES(`t1`.`int64_col`, 4)[safe_offset(2)] AS `q2`,
    APPROX_QUANTILES(`t1`.`int64_col`, 4)[safe_offset(3)] AS `q3`
  FROM (
    SELECT
      `t0`.`int64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`