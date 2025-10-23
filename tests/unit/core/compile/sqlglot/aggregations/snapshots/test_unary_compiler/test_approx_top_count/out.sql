WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64>>[STRUCT(CAST(NULL AS INT64))])
), `bfcte_1` AS (
  SELECT
    APPROX_TOP_COUNT(`bfcol_0`, 10) AS `bfcol_1`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `int64_col`
FROM `bfcte_1`