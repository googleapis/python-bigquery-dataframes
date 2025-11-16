SELECT
`float64_col` AS `float64_col`,
`int64_col` AS `int64_col`
FROM
(WITH `t1` AS (
  SELECT
    `t0`.`int64_col`,
    `t0`.`float64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
), `t2` AS (
  SELECT
    `t0`.`bool_col`,
    `t0`.`int64_too`,
    `t0`.`float64_col`
  FROM `bigframes-dev.sqlglot_test.scalar_types` AS `t0`
  WHERE
    `t0`.`bool_col`
)
SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      *
    FROM (
      SELECT
        `t9`.`float64_col`,
        `t9`.`int64_col`,
        `t9`.`bfuid_col_1161` AS `bfuid_col_1169`,
        `t9`.`bfuid_col_1160` AS `bfuid_col_1168`
      FROM (
        SELECT
          `t3`.`float64_col`,
          `t3`.`int64_col`,
          0 AS `bfuid_col_1161`,
          ROW_NUMBER() OVER (ORDER BY `t3`.`int64_col` IS NULL ASC, `t3`.`int64_col` ASC) - 1 AS `bfuid_col_1160`
        FROM `t1` AS `t3`
      ) AS `t9`
    ) AS `t11`
    UNION ALL
    SELECT
      *
    FROM (
      SELECT
        `t5`.`float64_col`,
        `t5`.`int64_too` AS `int64_col`,
        `t5`.`bfuid_col_1163` AS `bfuid_col_1169`,
        `t5`.`bfuid_col_1162` AS `bfuid_col_1168`
      FROM (
        SELECT
          `t4`.`float64_col`,
          `t4`.`int64_too`,
          1 AS `bfuid_col_1163`,
          ROW_NUMBER() OVER (ORDER BY NULL ASC) - 1 AS `bfuid_col_1162`
        FROM `t2` AS `t4`
      ) AS `t5`
    ) AS `t7`
  ) AS `t13`
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      *
    FROM (
      SELECT
        `t10`.`float64_col`,
        `t10`.`int64_col`,
        `t10`.`bfuid_col_1165` AS `bfuid_col_1169`,
        `t10`.`bfuid_col_1164` AS `bfuid_col_1168`
      FROM (
        SELECT
          `t3`.`float64_col`,
          `t3`.`int64_col`,
          2 AS `bfuid_col_1165`,
          ROW_NUMBER() OVER (ORDER BY `t3`.`int64_col` IS NULL ASC, `t3`.`int64_col` ASC) - 1 AS `bfuid_col_1164`
        FROM `t1` AS `t3`
      ) AS `t10`
    ) AS `t12`
    UNION ALL
    SELECT
      *
    FROM (
      SELECT
        `t6`.`float64_col`,
        `t6`.`int64_too` AS `int64_col`,
        `t6`.`bfuid_col_1167` AS `bfuid_col_1169`,
        `t6`.`bfuid_col_1166` AS `bfuid_col_1168`
      FROM (
        SELECT
          `t4`.`float64_col`,
          `t4`.`int64_too`,
          3 AS `bfuid_col_1167`,
          ROW_NUMBER() OVER (ORDER BY NULL ASC) - 1 AS `bfuid_col_1166`
        FROM `t2` AS `t4`
      ) AS `t6`
    ) AS `t8`
  ) AS `t14`
) AS `t15`)
ORDER BY `bfuid_col_1169` ASC NULLS LAST ,`bfuid_col_1168` ASC NULLS LAST