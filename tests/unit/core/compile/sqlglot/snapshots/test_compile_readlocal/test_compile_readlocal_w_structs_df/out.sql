SELECT `id`, `person` FROM (SELECT
  `t0`.`level_0` AS `id`,
  `t0`.`column_0` AS `person`,
  `t0`.`bfuid_col_1677` AS `bfuid_col_1678`
FROM (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`level_0` INT64, `column_0` STRUCT<name STRING, age INT64, address STRUCT<city STRING, country STRING>>, `bfuid_col_1677` INT64>>[STRUCT(
    1,
    STRUCT(
      'Alice' AS `name`,
      30 AS `age`,
      STRUCT('New York' AS `city`, 'USA' AS `country`) AS `address`
    ),
    0
  ), STRUCT(
    2,
    STRUCT(
      'Bob' AS `name`,
      25 AS `age`,
      STRUCT('London' AS `city`, 'UK' AS `country`) AS `address`
    ),
    1
  )]) AS `level_0`
) AS `t0`) AS `t`
ORDER BY `bfuid_col_1678` ASC NULLS LAST