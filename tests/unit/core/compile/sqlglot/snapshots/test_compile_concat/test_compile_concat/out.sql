SELECT
`rowindex` AS `rowindex`,
`rowindex_1` AS `rowindex_1`,
`int64_col` AS `int64_col`,
`string_col` AS `string_col`
FROM
(WITH `t1` AS (
  SELECT
    `t0`.`int64_col`,
    `t0`.`rowindex`,
    `t0`.`string_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
)
SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t3`.`bfuid_col_1` AS `rowindex`,
      `t3`.`rowindex` AS `rowindex_1`,
      `t3`.`int64_col`,
      `t3`.`string_col`,
      `t3`.`bfuid_col_1379` AS `bfuid_col_1383`,
      `t3`.`bfuid_col_1378` AS `bfuid_col_1382`
    FROM (
      SELECT
        `t2`.`rowindex` AS `bfuid_col_1`,
        `t2`.`rowindex`,
        `t2`.`int64_col`,
        `t2`.`string_col`,
        0 AS `bfuid_col_1379`,
        ROW_NUMBER() OVER (ORDER BY NULL ASC) - 1 AS `bfuid_col_1378`
      FROM `t1` AS `t2`
    ) AS `t3`
  ) AS `t5`
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      `t4`.`bfuid_col_1` AS `rowindex`,
      `t4`.`rowindex` AS `rowindex_1`,
      `t4`.`int64_col`,
      `t4`.`string_col`,
      `t4`.`bfuid_col_1381` AS `bfuid_col_1383`,
      `t4`.`bfuid_col_1380` AS `bfuid_col_1382`
    FROM (
      SELECT
        `t2`.`rowindex` AS `bfuid_col_1`,
        `t2`.`rowindex`,
        `t2`.`int64_col`,
        `t2`.`string_col`,
        1 AS `bfuid_col_1381`,
        ROW_NUMBER() OVER (ORDER BY NULL ASC) - 1 AS `bfuid_col_1380`
      FROM `t1` AS `t2`
    ) AS `t4`
  ) AS `t6`
) AS `t7`)
ORDER BY `bfuid_col_1383` ASC NULLS LAST ,`bfuid_col_1382` ASC NULLS LAST