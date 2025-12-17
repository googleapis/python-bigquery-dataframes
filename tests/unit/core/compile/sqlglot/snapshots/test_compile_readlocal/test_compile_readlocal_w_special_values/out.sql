WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` FLOAT64, `bfcol_1` FLOAT64, `bfcol_2` FLOAT64, `bfcol_3` FLOAT64, `bfcol_4` FLOAT64, `bfcol_5` BOOLEAN, `bfcol_6` INT64>>[STRUCT(
    CAST(NULL AS FLOAT64),
    CAST('Infinity' AS FLOAT64),
    CAST('-Infinity' AS FLOAT64),
    CAST(NULL AS FLOAT64),
    CAST(NULL AS FLOAT64),
    TRUE,
    0
  ), STRUCT(1.0, 1.0, 1.0, 1.0, 10.0, CAST(NULL AS BOOLEAN), 1), STRUCT(2.0, 2.0, 2.0, 2.0, 20.0, FALSE, 2)])
)
SELECT
  `bfcol_0` AS `col_none`,
  `bfcol_1` AS `col_inf`,
  `bfcol_2` AS `col_neginf`,
  `bfcol_3` AS `col_nan`,
  `bfcol_4` AS `col_int_none`,
  `bfcol_5` AS `col_bool_none`
FROM `bfcte_0`
ORDER BY
  `bfcol_6` ASC NULLS LAST