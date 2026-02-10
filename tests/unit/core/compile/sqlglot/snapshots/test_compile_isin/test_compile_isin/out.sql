SELECT
  `t1`.`bfuid_col_1` AS `rowindex`,
  EXISTS(
    SELECT
      1
    FROM (
      SELECT
        `t2`.`int64_too`
      FROM (
        SELECT
          `t0`.`int64_too`
        FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
      ) AS `t2`
      GROUP BY
        1
    ) AS `t3`
    WHERE
      (
        COALESCE(`t1`.`int64_col`, 0) = COALESCE(`t3`.`int64_too`, 0)
      )
      AND (
        COALESCE(`t1`.`int64_col`, 1) = COALESCE(`t3`.`int64_too`, 1)
      )
  ) AS `int64_col`
FROM (
  SELECT
    `t0`.`rowindex` AS `bfuid_col_1`,
    `t0`.`int64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
) AS `t1`