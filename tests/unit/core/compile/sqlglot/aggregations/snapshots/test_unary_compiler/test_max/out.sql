SELECT
  *
FROM (
  SELECT
    MAX(`t1`.`int64_col`) AS `int64_col`
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