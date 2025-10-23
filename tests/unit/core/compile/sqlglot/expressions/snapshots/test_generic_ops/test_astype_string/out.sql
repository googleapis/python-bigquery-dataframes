WITH `bfcte_0` AS (
  SELECT
    *
  FROM UNNEST(ARRAY<STRUCT<`bfcol_0` BOOLEAN, `bfcol_1` INT64, `bfcol_2` INT64>>[STRUCT(CAST(NULL AS BOOLEAN), CAST(NULL AS INT64), 0)])
), `bfcte_1` AS (
  SELECT
    *,
    CAST(`bfcol_1` AS STRING) AS `bfcol_3`,
    INITCAP(CAST(`bfcol_0` AS STRING)) AS `bfcol_4`,
    INITCAP(SAFE_CAST(`bfcol_0` AS STRING)) AS `bfcol_5`
  FROM `bfcte_0`
)
SELECT
  `bfcol_3` AS `int64_col`,
  `bfcol_4` AS `bool_col`,
  `bfcol_5` AS `bool_w_safe`
FROM `bfcte_1`
ORDER BY
  `bfcol_2` ASC NULLS LAST