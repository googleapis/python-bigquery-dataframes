SELECT
  `t2`.`bfuid_col_1` AS `rowindex`,
  EXISTS(
    SELECT
      1
    FROM (
      SELECT
        `t3`.`column_0`
      FROM (
        SELECT
          *
        FROM (
          SELECT
            *
          FROM UNNEST(ARRAY<STRUCT<`column_0` FLOAT64>>[STRUCT(314159.0), STRUCT(2.0), STRUCT(3.0), STRUCT(CAST(NULL AS FLOAT64))]) AS `column_0`
        ) AS `t1`
      ) AS `t3`
      GROUP BY
        1
    ) AS `t4`
    WHERE
      (
        COALESCE(`t2`.`int64_col`, 0) = COALESCE(`t4`.`column_0`, 0)
      )
      AND (
        COALESCE(`t2`.`int64_col`, 1) = COALESCE(`t4`.`column_0`, 1)
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
    FROM `bigframes-dev.sqlglot_test.scalar_types` FOR SYSTEM_TIME AS OF DATETIME('2025-08-26T20:51:00.718609')
  ) AS `t0`
) AS `t2`