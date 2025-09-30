SELECT
  *
FROM (
  SELECT
    approx_quantiles(`t1`.`int64_col`, 4)[safe_offset(1)] AS `q1`,
    approx_quantiles(`t1`.`int64_col`, 4)[safe_offset(2)] AS `q2`,
    approx_quantiles(`t1`.`int64_col`, 4)[safe_offset(3)] AS `q3`
  FROM (
    SELECT
      `t0`.`int64_col`
    FROM (
      SELECT
        `int64_col`
      FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
    ) AS `t0`
  ) AS `t1`
) AS `t2`