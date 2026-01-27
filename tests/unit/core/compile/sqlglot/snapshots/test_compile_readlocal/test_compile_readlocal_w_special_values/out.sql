SELECT
`col_none` AS `col_none`,
`col_inf` AS `col_inf`,
`col_neginf` AS `col_neginf`,
`col_nan` AS `col_nan`,
`col_struct_none` AS `col_struct_none`,
`col_struct_w_none` AS `col_struct_w_none`,
`col_list_none` AS `col_list_none`
FROM
(SELECT
  `t0`.`column_0` AS `col_none`,
  `t0`.`column_1` AS `col_inf`,
  `t0`.`column_2` AS `col_neginf`,
  `t0`.`column_3` AS `col_nan`,
  `t0`.`column_4` AS `col_struct_none`,
  `t0`.`column_5` AS `col_struct_w_none`,
  `t0`.`column_6` AS `col_list_none`,
  `t0`.`bfuid_col_1511` AS `bfuid_col_1512`
FROM (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`column_0` FLOAT64, `column_1` FLOAT64, `column_2` FLOAT64, `column_3` FLOAT64, `column_4` STRUCT<foo INT64>, `column_5` STRUCT<foo INT64>, `column_6` ARRAY<INT64>, `bfuid_col_1511` INT64>>[STRUCT(
    CAST(NULL AS FLOAT64),
    CAST('Infinity' AS FLOAT64),
    CAST('-Infinity' AS FLOAT64),
    CAST(NULL AS FLOAT64),
    CAST(NULL AS STRUCT<`foo` INT64>),
    STRUCT(CAST(NULL AS INT64) AS `foo`),
    ARRAY<INT64>[],
    0
  ), STRUCT(1.0, 1.0, 1.0, 1.0, STRUCT(1 AS `foo`), STRUCT(1 AS `foo`), [1, 2], 1), STRUCT(2.0, 2.0, 2.0, 2.0, STRUCT(2 AS `foo`), STRUCT(2 AS `foo`), [3, 4], 2)]) AS `column_0`
) AS `t0`)
ORDER BY `bfuid_col_1512` ASC NULLS LAST