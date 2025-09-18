SELECT
  `t2`.`bfuid_col_1` AS `rowindex`,
  EXISTS(
    SELECT
      1
    FROM (
      SELECT
        `t3`.`int64_too`
      FROM (
        SELECT
          `t1`.`int64_too`
        FROM (
          SELECT
            `int64_too`
          FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
        ) AS `t1`
      ) AS `t3`
      GROUP BY
        1
    ) AS `t4`
    WHERE
      (
        COALESCE(`t2`.`int64_col`, 0) = COALESCE(`t4`.`int64_too`, 0)
      )
      AND (
        COALESCE(`t2`.`int64_col`, 1) = COALESCE(`t4`.`int64_too`, 1)
      )
  ) AS `int64_col`
FROM (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1`,
    `t0`.`int64_col`
  FROM (
    SELECT
      `int64_col`,
      `rowindex`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
  ) AS `t0`
) AS `t2`