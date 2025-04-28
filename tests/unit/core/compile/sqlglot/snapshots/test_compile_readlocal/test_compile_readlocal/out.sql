SELECT
  *
FROM UNNEST(ARRAY<STRUCT<`column_0` INT64, `column_1` INT64, `column_2` BOOLEAN, `column_3` STRING, `bfuid_col_1` INT64>>[(1, -10, TRUE, 'b', 0), (2, 20, CAST(NULL AS BOOLEAN), 'aa', 1), (3, 30, FALSE, 'ccc', 2)])