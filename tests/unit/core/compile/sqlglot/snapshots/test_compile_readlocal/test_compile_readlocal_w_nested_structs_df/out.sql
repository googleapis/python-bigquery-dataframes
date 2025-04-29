SELECT
  *
FROM UNNEST(ARRAY<STRUCT<`level_0` INT64, `column_0` STRUCT<name STRING, age INT64, address STRUCT<city STRING, country STRING>>, `bfuid_col_3` INT64>>[(
  1,
  MAP(
    ['name', 'age', 'address'],
    ['Alice', 30, MAP(['city', 'country'], ['New York', 'USA'])]
  ),
  0
), (
  2,
  MAP(
    ['name', 'age', 'address'],
    ['Bob', 25, MAP(['city', 'country'], ['London', 'UK'])]
  ),
  1
)])