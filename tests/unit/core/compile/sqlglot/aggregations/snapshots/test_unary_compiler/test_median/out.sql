WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` DATE, `bfcol_1` INT64, `bfcol_2` STRING>>[STRUCT(CAST(NULL AS DATE), CAST(NULL AS INT64), CAST(NULL AS STRING))])
), `bfcte_1` AS (
  SELECT
    APPROX_QUANTILES(`bfcol_1`, 2)[OFFSET(1)] AS `bfcol_3`,
    APPROX_QUANTILES(`bfcol_0`, 2)[OFFSET(1)] AS `bfcol_4`,
    APPROX_QUANTILES(`bfcol_2`, 2)[OFFSET(1)] AS `bfcol_5`
  FROM `bfcte_0`
)
SELECT
  `bfcol_3` AS `int64_col`,
  `bfcol_4` AS `date_col`,
  `bfcol_5` AS `string_col`
FROM `bfcte_1`