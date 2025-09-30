SELECT
  *
FROM (
  SELECT
    COUNT(1) AS `size`
  FROM (
    SELECT
      `t0`.`rowindex` AS `bfuid_col_1`
    FROM (
      SELECT
        `rowindex`
      FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-30T20:19:48.854671')
    ) AS `t0`
  ) AS `t1`
) AS `t2`