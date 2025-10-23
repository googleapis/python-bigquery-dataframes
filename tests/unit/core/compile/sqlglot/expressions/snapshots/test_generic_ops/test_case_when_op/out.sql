WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` INT64, `bfcol_3` FLOAT64, `bfcol_4` INT64>>[STRUCT(
    CAST(NULL AS BOOLEAN),
    CAST(NULL AS INT64),
    CAST(NULL AS INT64),
    CAST(NULL AS FLOAT64),
    0
  )])
), `bfcte_1` AS (
  SELECT
    *,
    CASE WHEN `bfcol_0` THEN `bfcol_1` END AS `bfcol_5`,
    CASE WHEN `bfcol_0` THEN `bfcol_1` WHEN `bfcol_0` THEN `bfcol_2` END AS `bfcol_6`,
    CASE WHEN `bfcol_0` THEN `bfcol_0` WHEN `bfcol_0` THEN `bfcol_0` END AS `bfcol_7`,
    CASE
      WHEN `bfcol_0`
      THEN `bfcol_1`
      WHEN `bfcol_0`
      THEN CAST(`bfcol_0` AS INT64)
      WHEN `bfcol_0`
      THEN `bfcol_3`
    END AS `bfcol_8`
  FROM `bfcte_0`
)
SELECT
  `bfcol_5` AS `single_case`,
  `bfcol_6` AS `double_case`,
  `bfcol_7` AS `bool_types_case`,
  `bfcol_8` AS `mixed_types_cast`
FROM `bfcte_1`
ORDER BY
  `bfcol_4` ASC NULLS LAST