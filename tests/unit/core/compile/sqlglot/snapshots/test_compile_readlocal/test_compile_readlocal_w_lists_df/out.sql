SELECT
`rowindex` AS `rowindex`,
`int_list_col` AS `int_list_col`,
`bool_list_col` AS `bool_list_col`,
`float_list_col` AS `float_list_col`,
`date_list_col` AS `date_list_col`,
`date_time_list_col` AS `date_time_list_col`,
`numeric_list_col` AS `numeric_list_col`,
`string_list_col` AS `string_list_col`
FROM
(SELECT
  `t0`.`level_0` AS `rowindex`,
  `t0`.`column_0` AS `int_list_col`,
  `t0`.`column_1` AS `bool_list_col`,
  `t0`.`column_2` AS `float_list_col`,
  `t0`.`column_3` AS `date_list_col`,
  `t0`.`column_4` AS `date_time_list_col`,
  `t0`.`column_5` AS `numeric_list_col`,
  `t0`.`column_6` AS `string_list_col`,
  `t0`.`bfuid_col_1507` AS `bfuid_col_1508`
FROM (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`level_0` INT64, `column_0` ARRAY<INT64>, `column_1` ARRAY<BOOLEAN>, `column_2` ARRAY<FLOAT64>, `column_3` ARRAY<STRING>, `column_4` ARRAY<STRING>, `column_5` ARRAY<FLOAT64>, `column_6` ARRAY<STRING>, `bfuid_col_1507` INT64>>[STRUCT(
    0,
    [1],
    [TRUE],
    [1.2, 2.3],
    ['2021-07-21'],
    ['2021-07-21 11:39:45'],
    [1.2, 2.3, 3.4],
    ['abc', 'de', 'f'],
    0
  ), STRUCT(
    1,
    [1, 2],
    [TRUE, FALSE],
    [1.1],
    ['2021-07-21', '1987-03-28'],
    ['1999-03-14 17:22:00'],
    [5.5, 2.3],
    ['a', 'bc', 'de'],
    1
  ), STRUCT(
    2,
    [1, 2, 3],
    [TRUE],
    [0.5, -1.9, 2.3],
    ['2017-08-01', '2004-11-22'],
    ['1979-06-03 03:20:45'],
    [1.7000000000000002],
    ['', 'a'],
    2
  )]) AS `level_0`
) AS `t0`)
ORDER BY `bfuid_col_1508` ASC NULLS LAST