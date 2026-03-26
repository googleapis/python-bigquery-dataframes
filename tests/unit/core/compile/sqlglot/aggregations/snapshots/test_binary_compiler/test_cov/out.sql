SELECT
  *
FROM (
  SELECT
    COVAR_SAMP(`t1`.`int64_col`, `t1`.`float64_col`) AS `cov_col`
  FROM (
    SELECT
      `t0`.`int64_col`,
      `t0`.`float64_col`
    FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  ) AS `t1`
) AS `t2`