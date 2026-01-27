SELECT
  `t1`.`bfuid_col_1` AS `rowindex`,
  `t1`.`rowindex_2` IN (
    SELECT
      *
    FROM (
      SELECT
        `t2`.`rowindex_2`
      FROM (
        SELECT
          `t0`.`rowindex_2`
        FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
      ) AS `t2`
      GROUP BY
        1
    ) AS `t3`
  ) AS `rowindex_2`
FROM (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1`,
    `t0`.`rowindex_2`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`