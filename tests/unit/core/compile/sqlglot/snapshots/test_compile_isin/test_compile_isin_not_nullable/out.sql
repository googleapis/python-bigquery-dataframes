SELECT
  `t2`.`bfuid_col_1` AS `rowindex`,
  `t2`.`rowindex_2` IN (
    SELECT
      *
    FROM (
      SELECT
        `t3`.`rowindex_2`
      FROM (
        SELECT
          `t1`.`rowindex_2`
        FROM (
          SELECT
            `rowindex_2`
          FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
        ) AS `t1`
      ) AS `t3`
      GROUP BY
        1
    ) AS `t4`
  ) AS `rowindex_2`
FROM (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1`,
    `t0`.`rowindex_2`
  FROM (
    SELECT
      `rowindex`,
      `rowindex_2`
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-09-18T23:31:46.736473')
  ) AS `t0`
) AS `t2`