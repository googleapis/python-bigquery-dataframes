WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64, `bfcol_2` TIMESTAMP, `bfcol_3` INT64>>[STRUCT(CAST(NULL AS INT64), CAST(NULL AS INT64), CAST(NULL AS TIMESTAMP), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    `bfcol_1` AS `bfcol_8`,
    `bfcol_2` AS `bfcol_9`,
    `bfcol_0` AS `bfcol_10`,
    CAST(FLOOR(IEEE_DIVIDE(86400000000, `bfcol_0`)) AS INT64) AS `bfcol_11`
  FROM `bfcte_0`
)
SELECT
  `bfcol_8` AS `rowindex`,
  `bfcol_9` AS `timestamp_col`,
  `bfcol_10` AS `int64_col`,
  `bfcol_11` AS `timedelta_div_numeric`
FROM `bfcte_1`
ORDER BY
  `bfcol_3` ASC NULLS LAST