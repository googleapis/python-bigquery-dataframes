WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` ARRAY<INT64>, `bfcol_2` ARRAY<BOOLEAN>, `bfcol_3` ARRAY<FLOAT64>, `bfcol_4` ARRAY<DATE>, `bfcol_5` ARRAY<DATETIME>, `bfcol_6` ARRAY<NUMERIC>, `bfcol_7` ARRAY<STRING>, `bfcol_8` INT64>>[STRUCT(
    0,
    [1],
    [TRUE],
    [1.2, 2.3],
    [CAST('2021-07-21' AS DATE)],
    [CAST('2021-07-21T11:39:45' AS DATETIME)],
    [
      CAST(1.200000000 AS NUMERIC),
      CAST(2.300000000 AS NUMERIC),
      CAST(3.400000000 AS NUMERIC)
    ],
    ['abc', 'de', 'f'],
    0
  ), STRUCT(
    1,
    [1, 2],
    [TRUE, FALSE],
    [1.1],
    [CAST('2021-07-21' AS DATE), CAST('1987-03-28' AS DATE)],
    [CAST('1999-03-14T17:22:00' AS DATETIME)],
    [CAST(5.500000000 AS NUMERIC), CAST(2.300000000 AS NUMERIC)],
    ['a', 'bc', 'de'],
    1
  ), STRUCT(
    2,
    [1, 2, 3],
    [TRUE],
    [0.5, -1.9, 2.3],
    [CAST('2017-08-01' AS DATE), CAST('2004-11-22' AS DATE)],
    [CAST('1979-06-03T03:20:45' AS DATETIME)],
    [CAST(1.700000000 AS NUMERIC)],
    ['', 'a'],
    2
  )])
)
SELECT
  `bfcol_0` AS `rowindex`,
  `bfcol_1` AS `int_list_col`,
  `bfcol_2` AS `bool_list_col`,
  `bfcol_3` AS `float_list_col`,
  `bfcol_4` AS `date_list_col`,
  `bfcol_5` AS `date_time_list_col`,
  `bfcol_6` AS `numeric_list_col`,
  `bfcol_7` AS `string_list_col`
FROM `bfcte_0`
ORDER BY
  `bfcol_8` ASC NULLS LAST