WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` DATE, `bfcol_1` INT64, `bfcol_2` TIMESTAMP, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS DATE), CAST(NULL AS INT64), CAST(NULL AS TIMESTAMP), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    43200000000 AS `bfcol_8`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `rowindex`,
  `bfcol_2` AS `timestamp_col`,
  `bfcol_0` AS `date_col`,
  `bfcol_8` AS `timedelta_div_numeric`
FROM `bfcte_1`
ORDER BY
  `bfcol_3` ASC NULLS LAST