WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` STRING, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS STRING), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    CASE
      WHEN `bfcol_0` IS NULL
      THEN NULL
      ELSE COALESCE(LOGICAL_AND(`bfcol_0`) OVER (PARTITION BY `bfcol_1`), TRUE)
    END AS `bfcol_3`
  FROM `bfcte_0`
)
SELECT
  `bfcol_3` AS `agg_bool`
FROM `bfcte_1`
ORDER BY
  `bfcol_2` ASC NULLS LAST