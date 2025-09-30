SELECT
  *
FROM (
  SELECT
    approx_quantiles(`t1`.`int64_col`, 2)[offset(1)] AS `int64_col`,
    approx_quantiles(`t1`.`date_col`, 2)[offset(1)] AS `date_col`,
    approx_quantiles(`t1`.`string_col`, 2)[offset(1)] AS `string_col`
  FROM (
    SELECT
      `t0`.`date_col`,
      `t0`.`int64_col`,
      `t0`.`string_col`
    FROM (
      SELECT
        `date_col`,
        `int64_col`,
        `string_col`
      FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
    ) AS `t0`
  ) AS `t1`
) AS `t2`