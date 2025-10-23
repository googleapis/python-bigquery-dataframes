WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` INT64, `bfcol_1` INT64>>[STRUCT(CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    CASE WHEN `bfcol_0` IS NULL THEN NULL ELSE ANY_VALUE(`bfcol_0`) OVER () END AS `bfcol_2`
  FROM `bfcte_0`
)
SELECT
  `bfcol_2` AS `agg_int64`
FROM `bfcte_1`
ORDER BY
  `bfcol_1` ASC NULLS LAST