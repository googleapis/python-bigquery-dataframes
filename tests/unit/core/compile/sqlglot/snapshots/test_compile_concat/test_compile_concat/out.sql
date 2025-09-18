SELECT
`rowindex` AS `rowindex`,
`rowindex_1` AS `rowindex_1`,
`int64_col` AS `int64_col`,
`string_col` AS `string_col`
FROM
(SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      `t0`.`level_0` AS `rowindex`,
      `t0`.`column_9` AS `rowindex_1`,
      `t0`.`column_5` AS `int64_col`,
      `t0`.`column_11` AS `string_col`,
      0 AS `bfuid_col_792`,
      `t0`.`bfuid_col_785` AS `bfuid_col_791`
    FROM (
      SELECT
        *
      FROM UNNEST(ARRAY<STRUCT<`level_0` INT64, `column_5` INT64, `column_9` INT64, `column_11` STRING, `bfuid_col_785` INT64>>[STRUCT(0, 123456789, 0, 'Hello, World!', 0), STRUCT(1, -987654321, 1, 'こんにちは', 1), STRUCT(2, 314159, 2, '  ¡Hola Mundo!  ', 2), STRUCT(3, CAST(NULL AS INT64), 3, CAST(NULL AS STRING), 3), STRUCT(4, -234892, 4, 'Hello, World!', 4), STRUCT(5, 55555, 5, 'Güten Tag!', 5), STRUCT(6, 101202303, 6, 'capitalize, This ', 6), STRUCT(7, -214748367, 7, ' سلام', 7), STRUCT(8, 2, 8, 'T', 8)]) AS `level_0`
    ) AS `t0`
  ) AS `t1`
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      `t0`.`level_0` AS `rowindex`,
      `t0`.`column_9` AS `rowindex_1`,
      `t0`.`column_5` AS `int64_col`,
      `t0`.`column_11` AS `string_col`,
      1 AS `bfuid_col_792`,
      `t0`.`bfuid_col_785` AS `bfuid_col_791`
    FROM (
      SELECT
        *
      FROM UNNEST(ARRAY<STRUCT<`level_0` INT64, `column_5` INT64, `column_9` INT64, `column_11` STRING, `bfuid_col_785` INT64>>[STRUCT(0, 123456789, 0, 'Hello, World!', 0), STRUCT(1, -987654321, 1, 'こんにちは', 1), STRUCT(2, 314159, 2, '  ¡Hola Mundo!  ', 2), STRUCT(3, CAST(NULL AS INT64), 3, CAST(NULL AS STRING), 3), STRUCT(4, -234892, 4, 'Hello, World!', 4), STRUCT(5, 55555, 5, 'Güten Tag!', 5), STRUCT(6, 101202303, 6, 'capitalize, This ', 6), STRUCT(7, -214748367, 7, ' سلام', 7), STRUCT(8, 2, 8, 'T', 8)]) AS `level_0`
    ) AS `t0`
  ) AS `t2`
) AS `t3`)
ORDER BY `bfuid_col_792` ASC NULLS LAST ,`bfuid_col_791` ASC NULLS LAST