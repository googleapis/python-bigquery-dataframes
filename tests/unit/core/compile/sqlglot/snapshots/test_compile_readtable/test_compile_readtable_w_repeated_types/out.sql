WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` ARRAY<INT64>, `bfcol_2` ARRAY<BOOLEAN>, `bfcol_3` ARRAY<FLOAT64>, `bfcol_4` ARRAY<DATE>, `bfcol_5` ARRAY<DATETIME>, `bfcol_6` ARRAY<NUMERIC>, `bfcol_7` ARRAY<STRING>, `bfcol_8` INT64>>[STRUCT(
    CAST(NULL AS INT64),
    ARRAY<INT64>[],
    ARRAY<BOOLEAN>[],
    ARRAY<FLOAT64>[],
    ARRAY<DATE>[],
    ARRAY<DATETIME>[],
    ARRAY<NUMERIC>[],
    ARRAY<STRING>[],
    0
  )])
)
SELECT
  `bfcol_0` AS `rowindex`,
  `bfcol_0` AS `rowindex_1`,
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