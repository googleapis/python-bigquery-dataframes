WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64>>[STRUCT(CAST(NULL AS INT64))])
), `bfcte_1` AS (
  SELECT
    PERCENTILE_CONT(`bfcol_0`, 0.5) OVER () AS `bfcol_1`,
    CAST(FLOOR(PERCENTILE_CONT(`bfcol_0`, 0.5) OVER ()) AS INT64) AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_1` AS `quantile`,
  `bfcol_2` AS `quantile_floor`
FROM `bfcte_1`